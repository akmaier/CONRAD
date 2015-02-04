package edu.stanford.rsl.conrad.calibration;

import ij.ImagePlus;
import ij.gui.EllipseRoi;
import ij.gui.Overlay;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.gui.TextRoi;
import ij.process.FloatProcessor;
import ij.process.FloatStatistics;

import java.awt.Color;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import edu.mines.jtk.util.Array;
import edu.stanford.rsl.apps.gui.blobdetection.AutomaticMarkerDetectionWorker;
import edu.stanford.rsl.apps.gui.blobdetection.MarkerDetection;
import edu.stanford.rsl.apps.gui.blobdetection.MarkerDetectionWorker;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.FastRadialSymmetryTool;
import edu.stanford.rsl.conrad.filtering.NumericalDerivativeComputationTool;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.Rotations.BasicAxis;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.DecompositionQR;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.phantom.AbstractCalibrationPhantom;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.MathematicalPhantom;
import edu.stanford.rsl.conrad.phantom.RandomDistributionPhantom;
import edu.stanford.rsl.conrad.phantom.RandomizedHelixPhantom;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.hough.LineHoughSpace;

public class GeometricCalibration {

	/**
	 * configuration, storing the ideal trajectory
	 */
	Configuration config;

	/**
	 * storing ideal projection matrices derived from config
	 * 
	 * @see config
	 */
	Projection[] pMatricesIdeal;

	/**
	 * angle between principal components of ideal and detected beads
	 */
	double angle = 0.0;

	/**
	 * rotation matrix to eliminate the angle between ideal and detected beads
	 */
	SimpleMatrix rotation;

	/**
	 * detected beads
	 */
	ArrayList<PointND> beads2DUnsorted;

	/**
	 * ideally projected beads
	 */
	ArrayList<PointND> beads2DIdeal;

	/**
	 * formerly used for bead identification
	 * 
	 * @deprecated
	 */
	ArrayList<PointND> beads3Dbackprojected;

	/**
	 * formerly used for bead identification
	 * 
	 * @deprecated
	 */
	ArrayList<PointND> beads3DReal;

	/**
	 * correspondences between 2D and 3D coordinates of the beads
	 */
	ArrayList<CalibrationBead> cBeads;

	/**
	 * bead IDs stored according to cBeads
	 * 
	 * @see cBeads
	 */
	int[] ids;

	/**
	 * storing back projection error for each projection
	 */
	ArrayList<Double> errors;

	/**
	 * bead radius in px
	 */
	int beadRadius = 7;

	/**
	 * value to adjust the bead detection
	 */
	double threshold = 0.000;

	double largeBeadPercentileD = 0.999;
	double smallBeadPercentileD = 0.9998;
	double thresh1PercentileD = 0.000;
	double thresh2PercentileD = 0.05;

	/**
	 * minimal distance between to beads to be detected in px
	 */
	private double minimalDistanceBetweenTwoBeadsPx = 20;

	/**
	 * number of projection
	 */
	int slice = 0;

	/**
	 * calibration phantom name, inialized with RDP
	 */
	String phantomName = "Random Distribution Phantom";

	/**
	 * calibration phantom
	 */
	AbstractCalibrationPhantom phantom;

	/**
	 * storing the projection matrix for each projection
	 */
	ArrayList<SimpleMatrix> projectionMatrices = new ArrayList<SimpleMatrix>();

	/**
	 * constructor, initializes phantom, config, rotation, erros and projection
	 * matrices
	 * 
	 * @see phantom
	 * @see config
	 * @see rotation
	 * @see errors
	 * @see projectionMatrices
	 */
	public GeometricCalibration() {
		initPhantom();
		config = new Configuration();
		rotation = SimpleMatrix.I_4;
		errors = new ArrayList<Double>();
		projectionMatrices = new ArrayList<SimpleMatrix>();
	}

	/**
	 * method, to change phantom in the GUI
	 * 
	 * @param phantom
	 *            , name of the phantom
	 */
	public void setPhantom(String phantom) {
		this.phantomName = new String(phantom);
		initPhantom();
	}

	/**
	 * method, to adjust slice in the GUI
	 * 
	 * @param slice
	 */
	public void setSlice(int slice) {
		this.slice = slice;
	}

	/**
	 * initializes phantom according to phantomName
	 * 
	 * @see phantom
	 * @see phantomName
	 */
	private void initPhantom() {
		if (phantomName.equals("Random Distribution Phantom")) {
			phantom = new RandomDistributionPhantom();
		} else if (phantomName.equals("Randomized Helix Phantom")) {
			phantom = new RandomizedHelixPhantom();
		} else {
			phantom = new MathematicalPhantom();
		}
		phantom.init();

	}

	/**
	 * method to adjust detection in the GUI
	 * 
	 * @param rad
	 * @param thresh
	 */
	public void setDetection(int rad, double thresh) {
		beadRadius = rad;
		threshold = thresh;
	}

	/**
	 * 
	 * @param id
	 * @return 3D PointND containing x, y and z coordinates of the bead with id
	 *         id
	 */
	private PointND get3DCoordinates(int id) {
		return new PointND(phantom.getX(id), phantom.getY(id), phantom.getZ(id));
	}

	/**
	 * method to set config in the GUI
	 * 
	 * @param config
	 * @see config
	 * @deprecated
	 */
	public void setConfig(String config) {
		this.config = Configuration.loadConfiguration("d:\\Desktop\\" + config);
	}

	public void configure() {
		config = new Configuration();
		// pMatricesIdeal = ((CircularTrajectory) config.getGeometry())
		// .getProjectionMatrices();
		initPhantom();
		rotation = SimpleMatrix.I_4;
	}

	/**
	 * method to initialize pMatricesIdeal after loading a configuration in the
	 * GUI
	 */
	public void reConfigure() {
		pMatricesIdeal = ((CircularTrajectory) config.getGeometry())
				.getProjectionMatrices();
	}

	/**
	 * Computes the projection matrices using a list of correspondences
	 * 
	 * @param listCoords
	 * @return SimpleMatrix that maps world to image points
	 * @author Andreas Maier, Philipp Roser
	 */
	public static SimpleMatrix computePMatrix(
			ArrayList<CalibrationBead> listCoords) {

		Collections.sort(listCoords);
		int N = listCoords.size();
		// initialize measurement & dVector matrix
		SimpleMatrix measurements = new SimpleMatrix(2 * N, 11);
		SimpleMatrix dVector = new SimpleMatrix(2 * N, 1);

		for (int i = 0; i < N; i++) {
			CalibrationBead coords = listCoords.get(i);
			// 1st row of M
			measurements.setElementValue(i * 2, 0, coords.getX());
			measurements.setElementValue(i * 2, 1, coords.getY());
			measurements.setElementValue(i * 2, 2, coords.getZ());
			measurements.setElementValue(i * 2, 3, 1);
			measurements.setElementValue(i * 2, 4, 0);
			measurements.setElementValue(i * 2, 5, 0);
			measurements.setElementValue(i * 2, 6, 0);
			measurements.setElementValue(i * 2, 7, 0);
			measurements.setElementValue(i * 2, 8,
					-coords.getX() * coords.getU());
			measurements.setElementValue(i * 2, 9,
					-coords.getY() * coords.getU());
			measurements.setElementValue(i * 2, 10,
					-coords.getZ() * coords.getU());
			// 2nd row of M
			measurements.setElementValue(i * 2 + 1, 0, 0);
			measurements.setElementValue(i * 2 + 1, 1, 0);
			measurements.setElementValue(i * 2 + 1, 2, 0);
			measurements.setElementValue(i * 2 + 1, 3, 0);
			measurements.setElementValue(i * 2 + 1, 4, coords.getX());
			measurements.setElementValue(i * 2 + 1, 5, coords.getY());
			measurements.setElementValue(i * 2 + 1, 6, coords.getZ());
			measurements.setElementValue(i * 2 + 1, 7, 1);
			measurements.setElementValue(i * 2 + 1, 8,
					-coords.getX() * coords.getV());
			measurements.setElementValue(i * 2 + 1, 9,
					-coords.getY() * coords.getV());
			measurements.setElementValue(i * 2 + 1, 10,
					-coords.getZ() * coords.getV());

			// 1st & 2nd row of vector d
			dVector.setElementValue(i * 2, 0, coords.getU());
			dVector.setElementValue(i * 2 + 1, 0, coords.getV());
		}

		// compute pVector (11*1)
		DecompositionSVD svd = new DecompositionSVD(measurements);
		SimpleMatrix pVector = SimpleOperators.multiplyMatrixProd(
				svd.inverse(true), dVector);

		// rewrite to Matrix format 3 x 4 (p23 == 1)
		SimpleMatrix pMatrix = new SimpleMatrix(3, 4);
		pMatrix.setElementValue(0, 0, 700 * pVector.getElement(0, 0));
		pMatrix.setElementValue(0, 1, 700 * pVector.getElement(1, 0));
		pMatrix.setElementValue(0, 2, 700 * pVector.getElement(2, 0));
		pMatrix.setElementValue(0, 3, 700 * pVector.getElement(3, 0));
		pMatrix.setElementValue(1, 0, 700 * pVector.getElement(4, 0));
		pMatrix.setElementValue(1, 1, 700 * pVector.getElement(5, 0));
		pMatrix.setElementValue(1, 2, 700 * pVector.getElement(6, 0));
		pMatrix.setElementValue(1, 3, 700 * pVector.getElement(7, 0));
		pMatrix.setElementValue(2, 0, 700 * pVector.getElement(8, 0));
		pMatrix.setElementValue(2, 1, 700 * pVector.getElement(9, 0));
		pMatrix.setElementValue(2, 2, 700 * pVector.getElement(10, 0));
		pMatrix.setElementValue(2, 3, 700 * 1);
		return pMatrix;
	}

	/**
	 * estimates threshold
	 * 
	 * @param histogram
	 * @param stats
	 * @param percentile
	 * @return threshold
	 * @author Andreas Maier
	 */
	private float getHistogramThreshold(long[] histogram,
			FloatStatistics stats, double percentile) {
		int threshold = 0;
		long count = 0;
		for (threshold = 0; count < percentile * stats.pixelCount; threshold++) {
			count += histogram[threshold];
		}
		return (float) (threshold * stats.binSize + stats.histMin);
	}

	/**
	 * 
	 * @param linearHoughSpace
	 * @return
	 * @author Andreas Maier
	 */
	private int lineExtractBestLineCandidate(LineHoughSpace linearHoughSpace) {
		FloatProcessor houghspaceAsFloatProcessor = (FloatProcessor) linearHoughSpace
				.getImagePlus().getChannelProcessor();
		FloatStatistics stats = new FloatStatistics(houghspaceAsFloatProcessor);
		long[] histogram = stats.getHistogram();
		float edgeThreshold = getHistogramThreshold(histogram, stats, 0.9999);
		ArrayList<PointND> lineCandidate = General.extractCandidatePoints(
				houghspaceAsFloatProcessor, edgeThreshold); // 245
		ArrayList<PointND> lines = General.extractClusterCenter(lineCandidate,
				20); // distance
		return (int) lines.get(0).get(0);
	}

	/**
	 * 
	 * @param currentImage
	 * @param derivativeTool
	 * @param rightBorder
	 * @param leftBorder
	 * @return
	 * @author Andreas Maier
	 */
	protected Roi[] detectHoughLines(ImagePlus currentImage,
			NumericalDerivativeComputationTool derivativeTool, Roi rightBorder,
			Roi leftBorder) {
		// Compute derivative
		double houghLineThresh1 = 100;
		double houghLineThresh2 = 360;

		Grid2D before = ImageUtil.wrapImageProcessor(currentImage
				.getProcessor().toFloat(0, null));
		// /VisualizationUtil.showGrid2D(before, "before");

		// Configuration.getGlobalConfiguration().getGeometry()
		// .setPixelDimensionX(0.3);

		Grid2D derivative = derivativeTool.applyToolToImage(new Grid2D(before));

		for (int j = 0; j < derivative.getHeight(); j++) {
			for (int i = 0; i < derivative.getWidth(); i++) {
				derivative.setAtIndex(i, j,
						Math.abs(derivative.getPixelValue(i, j)));
			}
		}

		// VisualizationUtil.showGrid2D(derivative, "Deriv");

		FloatProcessor edgeImage = ImageUtil.wrapGrid2D(derivative);
		FloatStatistics stats = new FloatStatistics(edgeImage);
		long[] histogram = stats.getHistogram();
		houghLineThresh1 = getHistogramThreshold(histogram, stats, 0.95);
		houghLineThresh2 = getHistogramThreshold(histogram, stats, 0.99);
		LineHoughSpace houghLineLeft = new LineHoughSpace(1.0, 3.0,
				derivative.getWidth(), derivative.getHeight());
		LineHoughSpace houghLineRight = new LineHoughSpace(1.0, 3.0,
				derivative.getWidth(), derivative.getHeight());
		for (int j = 0; j < derivative.getHeight(); j++) {
			for (int i = 0; i < derivative.getWidth(); i++) {
				double value = Math.abs(derivative.getPixelValue(i, j));
				if (value > houghLineThresh1 && value < houghLineThresh2) { // thresholds
																			// for
																			// line
					if ((i > leftBorder.getBounds().x)
							&& (i < rightBorder.getBounds().x
									+ rightBorder.getBounds().width)) // set
																		// x-axis
																		// boundaries
																		// for
																		// outer
																		// cylinder
																		// line
						if (i < leftBorder.getBounds().x
								+ leftBorder.getBounds().width) {
							houghLineLeft.fill(i, j, 1.0); // cylinder boundary
															// line
						} else if (i > rightBorder.getBounds().x) {
							houghLineRight.fill(i, j, 1.0); // cylinder boundary
															// line
						}
				}
			}
		}

		// houghLineLeft.getImagePlus().show();
		// houghLineRight.getImagePlus().show();
		Roi leftLineRoi;
		Roi rightLineRoi;
		try {
			int leftLine = lineExtractBestLineCandidate(houghLineLeft);
			leftLineRoi = new Roi(leftLine, 0, 1, currentImage.getHeight());
			leftLineRoi.setName("LeftLineRoi");
			int rightLine = lineExtractBestLineCandidate(houghLineRight);
			rightLineRoi = new Roi(rightLine, 0, 1, currentImage.getHeight());
			rightLineRoi.setName("RightLineRoi");
		} catch (Exception e) {
			leftLineRoi = null;
			rightLineRoi = null;
		}

		return new Roi[] { leftLineRoi, rightLineRoi };
		// this happens in the GUI
		// overlayLines();

		// this.revalidate();
		// this.repaint();
	}

	/**
	 * 
	 * @param inputGrid
	 * @param radius
	 * @param percentile
	 * @param distance
	 * @param leftLineRoi
	 * @param rightLineRoi
	 * @return
	 * @author Andreas Maier
	 */
	protected ArrayList<PointND> extractBeads(Grid2D inputGrid, double radius,
			double percentile, double distance, Roi leftLineRoi,
			Roi rightLineRoi) {
		FastRadialSymmetryTool frst = new FastRadialSymmetryTool(new double[] {
				radius, radius + 2 }, 3, 0, null, 0);
		inputGrid = frst.applyToolToImage(new Grid2D(inputGrid));
		FloatProcessor floatP = ImageUtil.wrapGrid2D(inputGrid);

		FloatStatistics stats = new FloatStatistics(floatP);
		long[] histogram = stats.getHistogram();
		double thresh1 = getHistogramThreshold(histogram, stats, 0.000);
		double thresh2 = getHistogramThreshold(histogram, stats, 0.05);
		for (int j = 0; j < floatP.getHeight(); j++) {
			for (int i = 0; i < floatP.getWidth(); i++) {
				double value = (floatP.getPixelValue(i, j));
				if (value > thresh1 && value < thresh2) { // thresholds for line
					if ((i > leftLineRoi.getBounds().x)
							&& (i < rightLineRoi.getBounds().x)) // set x-axis
																	// boundaries
																	// for outer
																	// cylinder
																	// line
						floatP.putPixelValue(i, j, -floatP.getPixelValue(i, j));
					else {
						floatP.putPixelValue(i, j, 0.0);
					}
				} else
					floatP.putPixelValue(i, j, 0.0);
			}
		}

		stats = new FloatStatistics(floatP);
		histogram = stats.getHistogram();
		thresh1 = getHistogramThreshold(histogram, stats, percentile);

		ArrayList<PointND> candidate = General.extractCandidatePoints(floatP,
				thresh1);
		// filter with min distance
		return General.extractClusterCenter(candidate, distance, false); // distance
	}

	/**
	 * 
	 * @param tmpBead
	 * @param pMatrix
	 * @return
	 * @author Andreas Maier
	 */
	public static double[] compute2DCoordinates(CalibrationBead tmpBead,
			SimpleMatrix pMatrix) {
		SimpleMatrix Q3D = new SimpleMatrix(4, 1);
		Q3D.setElementValue(0, 0, tmpBead.getX());
		Q3D.setElementValue(1, 0, tmpBead.getY());
		Q3D.setElementValue(2, 0, tmpBead.getZ());
		Q3D.setElementValue(3, 0, 1);
		SimpleMatrix Q2D = SimpleOperators.multiplyMatrixProd(pMatrix, Q3D);
		return new double[] { Q2D.getElement(0, 0) / Q2D.getElement(2, 0),
				Q2D.getElement(1, 0) / Q2D.getElement(2, 0) };
	}

	/**
	 * 
	 * @param point3D
	 * @param pMatrix
	 * @param slice
	 * @return
	 * @author Martin Berger
	 */
	private double[] compute2Dfrom3D(double[] point3D, Projection pMatrix,
			int slice) {

		// Compute coordinates in projection data.
		SimpleVector homogeneousPoint = SimpleOperators.multiply(pMatrix
				.computeP(), new SimpleVector(point3D[0], point3D[1],
				point3D[2], 1));
		// Do forward projection to 2D coordinates
		double coordU = homogeneousPoint.getElement(0)
				/ homogeneousPoint.getElement(2);
		double coordV = homogeneousPoint.getElement(1)
				/ homogeneousPoint.getElement(2);

		return new double[] { coordU, coordV, (double) slice };
	}

	/**
	 * 
	 * @param point2D
	 * @param p
	 * @return
	 * @author Martin Berger
	 */
	private PointND compute3Dfrom2D(PointND point2D, Projection p) {
		SimpleVector homP = SimpleOperators.multiply(
				p.computeP().inverse(InversionType.INVERT_SVD),
				new SimpleVector(point2D.get(0), point2D.get(1), 1));
		double x = homP.getElement(0) / homP.getElement(3);
		double y = homP.getElement(1) / homP.getElement(3);
		double z = homP.getElement(2) / homP.getElement(3);
		return new PointND(x, y, z);
	}

	protected double measureDistance(int i, PointND p1, PointND p2) {
		if (i < 1 || i > 3) {
			throw new IllegalArgumentException();
		} else if (i == 1) {
			return Math.min(
					p1.euclideanDistance(p2),
					Math.abs(p1.get(p1.getDimension() - 1)
							- p2.get(p2.getDimension() - 1)));
		} else {
			return p1.euclideanDistance(p2);
		}
	}

	/**
	 * method only for test issues
	 * 
	 * @deprecated
	 */
	protected void computeBeadIDsBack() {

		// initialize array storing all euclidean distances between real and
		// real 3D positions
		double[][] distMeasure = new double[beads3Dbackprojected.size()][beads3DReal
				.size()];

		int r = 0;
		int i = 0;

		// actual measurements
		for (PointND real : beads3Dbackprojected) {
			for (PointND ideal : beads3DReal) {
				double dist = measureDistance(2, real, ideal);
				distMeasure[r][i] = dist;
				i++;
			}
			i = 0;
			r++;
		}

		// initalize arrays for potential minima and corresponding bead ids
		double[] potentialMinima = new double[beads3Dbackprojected.size()];
		int[] potentialIds = new int[beads3Dbackprojected.size()];

		// find minimum for each real 2D position detected
		for (r = 0; r < beads3Dbackprojected.size(); r++) {
			double min = Array.max(distMeasure[r]);
			for (i = 0; i < beads3DReal.size(); i++) {
				if (distMeasure[r][i] < min) {
					min = distMeasure[r][i];
					potentialIds[r] = i;
				}
			}
			potentialMinima[r] = min;
		}

		// initialize list and arrays for actual calibration
		cBeads = new ArrayList<CalibrationBead>();
		CalibrationBead[] minima = new CalibrationBead[beads3Dbackprojected
				.size()];
		ids = new int[beads3Dbackprojected.size()];

		// find n = 8 correspondences with smallest euclidean distance
		double max = Array.max(potentialMinima);
		for (int n = 0; n < beads3Dbackprojected.size(); n++) {
			double min = Array.max(potentialMinima);
			int id = 0;
			int pos = 0;
			for (r = 0; r < potentialMinima.length; r++) {

				if (potentialMinima[r] <= min) {
					min = potentialMinima[r];
					id = potentialIds[r];
					pos = r;
				}
			}
			potentialMinima[pos] = max;
			ids[n] = id;
			minima[n] = new CalibrationBead(beads2DUnsorted.get(pos).get(0),
					beads2DUnsorted.get(pos).get(1));
			phantom.setBeadCoordinates(minima[n], ids[n]);
			cBeads.add(minima[n]);
		}

	}

	// only for test issues
	/**
	 * method only for test issues
	 * 
	 * @deprecated
	 */
	protected void computeBeadIDsIdeal() {
		// initialize array storing all euclidean distances between real and
		// ideal 2D positions
		double[][] distMeasure = new double[beads2DIdeal.size()][beads2DIdeal
				.size()];
		int r = 0;
		int i = 0;

		// actual measurements
		for (PointND real : beads2DIdeal) {
			for (PointND ideal : beads2DIdeal) {
				double dist = measureDistance(2, real, ideal);
				distMeasure[r][i] = dist;
				i++;
			}
			i = 0;
			r++;
		}

		// initalize arrays for potential minima and corresponding bead ids
		double[] potentialMinima = new double[beads2DIdeal.size()];
		int[] potentialIds = new int[beads2DIdeal.size()];

		// find minimum for each real 2D position detected
		for (r = 0; r < beads2DIdeal.size(); r++) {
			double min = 1000;
			for (i = 0; i < beads2DIdeal.size(); i++) {
				if (distMeasure[r][i] < min) {
					min = distMeasure[r][i];
					potentialIds[r] = i;
				}
			}
			potentialMinima[r] = min;
		}

		// initialize list and arrays for actual calibration
		cBeads = new ArrayList<CalibrationBead>();
		CalibrationBead[] minima = new CalibrationBead[beads2DIdeal.size()];
		ids = new int[beads2DIdeal.size()];

		// find n = 8 correspondences with smallest euclidean distance
		double max = 1000;
		for (int n = 0; n < beads2DIdeal.size(); n++) {
			double min = 1000;
			int id = 0;
			int pos = 0;
			for (r = 0; r < potentialMinima.length; r++) {

				if (potentialMinima[r] <= min) {
					min = potentialMinima[r];
					id = potentialIds[r];
					pos = r;
				}
			}
			potentialMinima[pos] = max;
			ids[n] = id;
			minima[n] = new CalibrationBead(beads2DIdeal.get(pos).get(0),
					beads2DIdeal.get(pos).get(1));
			phantom.setBeadCoordinates(minima[n], ids[n]);
			cBeads.add(minima[n]);
		}

	}

	/**
	 * estimates rotation between ideally projected and actually detected beads
	 * using PCA
	 * 
	 * @see PrincipalComponentAnalysis2D
	 */
	protected void estimateRotation() {
		PrincipalComponentAnalysis2D real = new PrincipalComponentAnalysis2D(
				beads2DUnsorted);
		PrincipalComponentAnalysis2D ideal = new PrincipalComponentAnalysis2D(
				beads2DIdeal);
		System.out.println("Real EV:" + real.stringEigenvectors());
		System.out.println("Ideal EV:" + ideal.stringEigenvectors());
		Matrix realV = real.getEigenvectors()[0]
				.plus(real.getEigenvectors()[1]);
		Matrix idealV = ideal.getEigenvectors()[0]
				.plus(ideal.getEigenvectors()[1]);
		double scalar = Math.abs(realV.get(0, 0) * idealV.get(0, 0))
				+ Math.abs(realV.get(1, 0) * idealV.get(1, 0));
		angle = Math.acos(scalar / (realV.norm2() * idealV.norm2()));
		System.out.println("Angle = " + angle * 180.0 / Math.PI);
		double s = Math.sin(angle);
		double c = Math.cos(angle);
		rotation = new SimpleMatrix(
				new double[][] { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, c, -s, 0.0 },
						{ 0.0, s, c, 0.0 }, { 0.0, 0.0, 0.0, 1.0 } });
	}

	// compares real and ideal 2D positions to establish most probable
	// correspondences
	/**
	 * establishes correspondences between world and image points by linking
	 * ideal and detected beads via the euclidean distance
	 */
	protected void computeBeadIDsForward() {

		beads2DIdeal = new ArrayList<PointND>();
		for (int i = 0; i < numberOfBeads(); i++) {
			double[] coord = compute2Dfrom3D(
					new double[] { get3DCoordinates(i).get(0),
							get3DCoordinates(i).get(1),
							get3DCoordinates(i).get(2) },
					new Projection(SimpleOperators.multiplyMatrixProd(
							pMatricesIdeal[slice].computeP(), rotation)), slice);
			beads2DIdeal.add(i, new PointND(coord[0], coord[1]));
		}

		// initialize array storing all euclidean distances between real and
		// ideal 2D positions
		double[][] distMeasure = new double[beads2DUnsorted.size()][beads2DIdeal
				.size()];
		int r = 0;
		int i = 0;

		// actual measurements
		for (PointND real : beads2DUnsorted) {
			for (PointND ideal : beads2DIdeal) {
				double dist = real.euclideanDistance(ideal);
				distMeasure[r][i] = dist;
				i++;
			}
			i = 0;
			r++;
		}

		// initalize arrays for potential minima and corresponding bead ids
		double[] potentialMinima = new double[beads2DUnsorted.size()];
		int[] potentialIds = new int[beads2DUnsorted.size()];

		// find minimum for each real 2D position detected
		for (r = 0; r < beads2DUnsorted.size(); r++) {
			double min = Array.max(distMeasure[r]);
			for (i = 0; i < beads2DIdeal.size(); i++) {
				if (distMeasure[r][i] < min) {
					min = distMeasure[r][i];
					potentialIds[r] = i;
				}
			}
			potentialMinima[r] = min;
		}

		// initialize list and arrays for actual calibration
		cBeads = new ArrayList<CalibrationBead>();
		CalibrationBead[] minima = new CalibrationBead[beads2DUnsorted.size()];
		ids = new int[beads2DUnsorted.size()];

		// find n = 8 correspondences with smallest euclidean distance
		double max = Array.max(potentialMinima) + 1;
		for (int n = 0; n < beads2DUnsorted.size(); n++) {
			double min = Array.max(potentialMinima) + 1;
			int id = 0;
			int pos = 0;
			for (r = 0; r < potentialMinima.length; r++) {

				if (potentialMinima[r] <= min) {
					min = potentialMinima[r];
					id = potentialIds[r];
					pos = r;
				}
			}
			potentialMinima[pos] = max;

			ids[n] = id;
			CalibrationBead cb = new CalibrationBead(beads2DUnsorted.get(pos)
					.get(0), beads2DUnsorted.get(pos).get(1));
			minima[n] = new CalibrationBead(beads2DUnsorted.get(pos).get(0),
					beads2DUnsorted.get(pos).get(1));
			phantom.setBeadCoordinates(minima[n], ids[n]);
			phantom.setBeadCoordinates(cb, id);
			cBeads.add(cb);
		}

	}

	/**
	 * contains all image points of all frames
	 */
	ArrayList<ArrayList<PointND>> imagePoints;

	/**
	 * reference points to sort newly detected beads
	 */
	ArrayList<PointND> referencePoints;

	/**
	 * potential bead ids to be removed
	 */
	ArrayList<Integer> remove;
	int loc = 0;

	PointND[][] image;
	PointND[] reference;

	/**
	 * Finds initial points for Factorization. Currently configured for
	 * debugging.
	 * 
	 * @param pointSelector
	 *            , if true initial points are estimated manually
	 * @param im
	 * @param leftLineRoi
	 * @param rightLineRoi
	 */
	protected void getInitialPoints(boolean pointSelector, ImagePlus im,
			Roi leftLineRoi, Roi rightLineRoi) {
		referencePoints = new ArrayList<PointND>();
		imagePoints = new ArrayList<ArrayList<PointND>>();
		remove = new ArrayList<Integer>();
		if (pointSelector) {

		} else {
			// detectBeads(im, leftLineRoi, rightLineRoi);
			// referencePoints = (ArrayList<PointND>) beads2DUnsorted.clone();
			// imagePoints.add(0, (ArrayList<PointND>) referencePoints.clone());
			idealBeads();
			imagePoints.add(new ArrayList<PointND>(beads2DIdeal));
		}
	}

	/**
	 * method for test issues
	 * 
	 * @param pointSelector
	 * @param im
	 * @param leftLineRoi
	 * @param rightLineRoi
	 * @deprecated
	 */
	protected void getInitialPointsArray(boolean pointSelector, ImagePlus im,
			Roi leftLineRoi, Roi rightLineRoi) {
		reference = new PointND[numberOfBeads()];
		image = new PointND[im.getImageStackSize()][numberOfBeads()];
		remove = new ArrayList<Integer>();
		if (pointSelector) {

		} else {
			detectBeads(im, leftLineRoi, rightLineRoi);
			for (int i = 0; i < reference.length; i++) {
				reference[i] = beads2DUnsorted.get(i);
			}
			image[0] = reference;
		}
	}

	/**
	 * Establishes correspondences via linking detected beads to former
	 * established correspondences
	 * 
	 * @param im
	 * @param leftLineRoi
	 * @param rightLineRoi
	 */
	protected void findPointsNaive(ImagePlus im, Roi leftLineRoi,
			Roi rightLineRoi) {

		ArrayList<CalibrationBead> newC = new ArrayList<CalibrationBead>();

		for (CalibrationBead cb : cBeads) {
			boolean found = false;

			int pos = -1;
			PointND ref = new PointND(cb.getU(), cb.getV());
			double thresh = 15.0;
			for (PointND det : beads2DUnsorted) {

				CalibrationBead newBead = new CalibrationBead(0.0, 0.0);
				newC.add(newBead);
				double dist = ref.euclideanDistance(det);

				if (dist < thresh) {
					thresh = dist;
					newBead.setU(det.get(0));
					newBead.setV(det.get(1));
					newBead.setX(cb.getX());
					newBead.setY(cb.getY());
					newBead.setZ(cb.getZ());
					found = true;
					newC.add(newBead);
					pos = beads2DUnsorted.indexOf(det);
				}
			}

			if (!found) {
				newC.add(cb);
			} else {
				beads2DUnsorted.remove(pos);
			}

		}
		cBeads = new ArrayList<CalibrationBead>(newC);
	}

	/**
	 * establishes point sets for Factorization. Configured for debugging
	 * 
	 * @param im
	 * @param leftLineRoi
	 * @param rightLineRoi
	 */
	protected void findPoints(ImagePlus im, Roi leftLineRoi, Roi rightLineRoi) {

		beads2DIdeal = new ArrayList<PointND>();
		idealBeads();
		ArrayList<PointND> newL = new ArrayList<PointND>(beads2DIdeal);
		imagePoints.add(newL);
		/*
		 * beads2DUnsorted = new ArrayList<PointND>(); detectBeads(im,
		 * leftLineRoi, rightLineRoi); ArrayList<PointND> list = new
		 * ArrayList<PointND>(numberOfBeads()); int size = 0;
		 * 
		 * for (PointND ref : referencePoints) {
		 * 
		 * boolean found = false; int pos = referencePoints.indexOf(ref);
		 * 
		 * if (ref == null) { continue; }
		 * 
		 * for (PointND det : beads2DUnsorted) {
		 * 
		 * if (Math.abs(ref.get(1) - det.get(1)) < 15.0) { if
		 * (list.contains(det)) { if (found &&
		 * list.get(pos).euclideanDistance(ref) > det .euclideanDistance(ref)) {
		 * list.add(pos, det); } } if (!found && !list.contains(det)) {
		 * list.add(pos, det); found = true; size++; } } }
		 * 
		 * if (!found) { list.add(pos, referencePoints.get(pos)); remove.add(new
		 * Integer(pos)); }
		 * 
		 * } System.out.println("Slice: " + slice + ", beads detected: " +
		 * beads2DUnsorted.size() + ", beads found: " + size);
		 * System.out.println(list); imagePoints.add(list); referencePoints =
		 * list;
		 */
	}

	/**
	 * still testing
	 */
	protected void removePoints() {
		remove = new ArrayList<Integer>();
		for (Integer i : remove) {
			for (ArrayList<PointND> list : imagePoints) {
				list.remove(remove.get(i).intValue());
			}
		}
	}

	/**
	 * computes Factorization
	 */
	protected void factorize() {
		Factorization fac = new Factorization(imagePoints);
		projectionMatrices = fac.getProjectionMatrices();
	}

	/**
	 * 
	 * @return number of beads of the used phantom
	 */
	private int numberOfBeads() {
		return phantom.getNumberOfBeads();
	}

	// sorted IDs
	/**
	 * computes ideal beads
	 */
	protected void idealBeads() {
		beads2DIdeal = new ArrayList<PointND>();
		for (int i = 0; i < numberOfBeads(); i++) {
			double[] coord = compute2Dfrom3D(new double[] {
					get3DCoordinates(i).get(0), get3DCoordinates(i).get(1),
					get3DCoordinates(i).get(2) }, pMatricesIdeal[slice], slice);
			beads2DIdeal.add(i, new PointND(coord[0], coord[1]));
		}
	}

	// unsorted IDs
	/**
	 * detect beads using FRST
	 * 
	 * @param im
	 * @param leftLineRoi
	 * @param rightLineRoi
	 */
	protected void detectBeads(ImagePlus im, Roi leftLineRoi, Roi rightLineRoi) {
		beads2DUnsorted = new ArrayList<PointND>();
		beads2DUnsorted = extractBeads(ImageUtil.wrapImageProcessor(im
				.getProcessor().toFloat(0, null)), beadRadius,
				smallBeadPercentileD, minimalDistanceBetweenTwoBeadsPx,
				leftLineRoi, rightLineRoi);
	}

	/**
	 * computes backprojected beads
	 * 
	 * @deprecated
	 */
	protected void backprojectBeads() {
		beads3Dbackprojected = new ArrayList<PointND>();
		int i = 0;
		for (PointND p2D : beads2DUnsorted) {
			beads3Dbackprojected.add(i,
					compute3Dfrom2D(p2D, pMatricesIdeal[slice]));
			i++;
		}
	}

	/**
	 * provides 3D coordinates of the beads
	 */
	protected void real3DBeads() {
		beads3DReal = new ArrayList<PointND>();
		for (int i = 0; i < numberOfBeads(); i++) {
			beads3DReal.add(i, get3DCoordinates(i));
		}
	}

	// created only for test issues
	/**
	 * only for test issues
	 * 
	 * @deprecated
	 * @return
	 */
	protected SimpleMatrix computePMatrixIdeal() {
		cBeads = new ArrayList<CalibrationBead>();
		for (int i = 0; i < beads3DReal.size(); i++) {
			CalibrationBead cb = new CalibrationBead(
					beads2DIdeal.get(i).get(0), beads2DIdeal.get(i).get(1));
			cb.setX(beads3DReal.get(i).get(0));
			cb.setY(beads3DReal.get(i).get(1));
			cb.setZ(beads3DReal.get(i).get(2));
			cBeads.add(cb);
		}
		return computePMatrix(cBeads);
	}

	/**
	 * calibration using factorization, not working completely yet
	 * 
	 * @param sliceNumber
	 * @param overlay
	 * @return back projection error
	 * @author Andreas Maier, Philipp Roser
	 */
	public double calibrateF(int sliceNumber, Overlay overlay) {
		SimpleMatrix pMatrix = projectionMatrices.get(sliceNumber);
		System.out.println("slice" + sliceNumber + ": " + pMatrix);
		double backprojectionError = 0;

		for (int i = 0; i < numberOfBeads(); i++) {
			CalibrationBead tmpBead = new CalibrationBead(0, 0);
			phantom.setBeadCoordinates(tmpBead, i);

			double[] coord2d = GeometricCalibration.compute2DCoordinates(
					tmpBead, pMatrix);

			for (PointND b : beads2DIdeal) {
				PointND p = new PointND(coord2d[0], coord2d[1]);
				if (b.euclideanDistance(p) < CONRAD.FLOAT_EPSILON) {
					backprojectionError += b.euclideanDistance(p);
				}
			}

			// mark projected bead in image.
			overlay.add(new PointRoi(coord2d[0], coord2d[1]));
			NumberFormat nf = NumberFormat.getInstance();
			nf.setMaximumFractionDigits(2);
			nf.setMinimumFractionDigits(2);
			TextRoi text = new TextRoi(coord2d[0], coord2d[1] - 20, "Bead " + i);
			overlay.add(text);

		}

		backprojectionError /= beads2DUnsorted.size();

		// if(backprojectionError > 1.0){
		// System.out.println("Error: "+ backprojectionError);
		// }

		// currentImage.setOverlay(overlay);
		// config.getGeometry().getProjectionMatrices()[sliceNumber] = new
		// Projection(
		// new SimpleMatrix(pMatrix));
		/**
		 * Focal spot Sw = -inverse of M * P3
		 */
		SimpleMatrix M = new SimpleMatrix(3, 3);
		SimpleMatrix Sw = new SimpleMatrix(3, 1);
		SimpleMatrix P3 = new SimpleMatrix(3, 1);

		M.setElementValue(0, 0, pMatrix.getElement(0, 0));
		M.setElementValue(0, 1, pMatrix.getElement(0, 1));
		M.setElementValue(0, 2, pMatrix.getElement(0, 2));
		M.setElementValue(1, 0, pMatrix.getElement(1, 0));
		M.setElementValue(1, 1, pMatrix.getElement(1, 1));
		M.setElementValue(1, 2, pMatrix.getElement(1, 2));
		M.setElementValue(2, 0, pMatrix.getElement(2, 0));
		M.setElementValue(2, 1, pMatrix.getElement(2, 1));
		M.setElementValue(2, 2, pMatrix.getElement(2, 2));
		P3.setElementValue(0, 0, pMatrix.getElement(0, 3));
		P3.setElementValue(1, 0, pMatrix.getElement(1, 3));
		P3.setElementValue(2, 0, pMatrix.getElement(2, 3));

		Sw = SimpleOperators.multiplyMatrixProd(
				M.inverse(SimpleMatrix.InversionType.INVERT_QR), P3)
				.multipliedBy(-1.0);
		// primary angle
		double priAngle = Math.atan(Math.abs(Sw.getElement(1, 0)
				/ Sw.getElement(0, 0)));
		if (Sw.getElement(0, 0) < 0 && Sw.getElement(1, 0) >= 0)
			priAngle = Math.PI - priAngle;// in 2nd quadrant
		else if (Sw.getElement(0, 0) < 0 && Sw.getElement(1, 0) < 0)
			priAngle = Math.PI + priAngle;// in 3rd quadrant
		else if (Sw.getElement(0, 0) >= 0 && Sw.getElement(1, 0) < 0)
			priAngle = 2 * Math.PI - priAngle;// in 4th quadrant

		if (priAngle >= Math.PI)
			priAngle = -(Math.PI * 2 - priAngle);

		priAngle *= 180.0 / Math.PI;

		// secondary angle
		double sndAngle = Math.atan(Sw.getElement(2, 0)
				/ Math.sqrt(Math.pow(Sw.getElement(0, 0), 2)
						+ Math.pow(Sw.getElement(1, 0), 2)));

		sndAngle *= 180.0 / Math.PI;

		// config.getGeometry().getPrimaryAngles()[sliceNumber] = priAngle;
		// config.getGeometry().getSecondaryAngles()[sliceNumber] = sndAngle;
		// this.revalidate();
		// this.repaint();
		return backprojectionError;
	}

	/**
	 * calibration using correspondences
	 * 
	 * @param sliceNumber
	 * @param overlay
	 * @return back projection error
	 * @author Andreas Maier, Philipp Roser
	 */
	public double calibrate(int sliceNumber, Overlay overlay) {

		SimpleMatrix pMatrix = GeometricCalibration.computePMatrix(cBeads);
		projectionMatrices.add(pMatrix);
		double backprojectionError = 0;

		for (int i = 0; i < numberOfBeads(); i++) {
			CalibrationBead tmpBead = new CalibrationBead(0, 0);
			phantom.setBeadCoordinates(tmpBead, i);

			double[] coord2d = GeometricCalibration.compute2DCoordinates(
					tmpBead, pMatrix);

			// backprojectionError += beads2DIdeal.get(i).euclideanDistance(new
			// PointND(coord2d[0], coord2d[1]));

			for (CalibrationBead b : cBeads) {
				PointND threeDPoint = new PointND(b.getX(), b.getY(), b.getZ());
				PointND threeDPoint2 = new PointND(tmpBead.getX(),
						tmpBead.getY(), tmpBead.getZ());
				if (threeDPoint.euclideanDistance(threeDPoint2) < CONRAD.FLOAT_EPSILON) {
					PointND twoDPoint = new PointND(b.getU(), b.getV());
					PointND twoDPoint2 = new PointND(coord2d[0], coord2d[1]);
					backprojectionError += twoDPoint
							.euclideanDistance(twoDPoint2);
				}
			}

			// mark projected bead in image.
			overlay.add(new PointRoi(coord2d[0], coord2d[1]));
			NumberFormat nf = NumberFormat.getInstance();
			nf.setMaximumFractionDigits(2);
			nf.setMinimumFractionDigits(2);
			TextRoi text = new TextRoi(coord2d[0], coord2d[1] - 20, "Bead " + i);
			overlay.add(text);

		}

		backprojectionError /= cBeads.size();

		// if(backprojectionError > 1.0){
		// System.out.println("Error: "+ backprojectionError);
		// }

		// currentImage.setOverlay(overlay);
		// config.getGeometry().getProjectionMatrices()[sliceNumber] = new
		// Projection(
		// new SimpleMatrix(pMatrix));
		/**
		 * Focal spot Sw = -inverse of M * P3
		 */
		SimpleMatrix M = new SimpleMatrix(3, 3);
		SimpleMatrix Sw = new SimpleMatrix(3, 1);
		SimpleMatrix P3 = new SimpleMatrix(3, 1);

		M.setElementValue(0, 0, pMatrix.getElement(0, 0));
		M.setElementValue(0, 1, pMatrix.getElement(0, 1));
		M.setElementValue(0, 2, pMatrix.getElement(0, 2));
		M.setElementValue(1, 0, pMatrix.getElement(1, 0));
		M.setElementValue(1, 1, pMatrix.getElement(1, 1));
		M.setElementValue(1, 2, pMatrix.getElement(1, 2));
		M.setElementValue(2, 0, pMatrix.getElement(2, 0));
		M.setElementValue(2, 1, pMatrix.getElement(2, 1));
		M.setElementValue(2, 2, pMatrix.getElement(2, 2));
		P3.setElementValue(0, 0, pMatrix.getElement(0, 3));
		P3.setElementValue(1, 0, pMatrix.getElement(1, 3));
		P3.setElementValue(2, 0, pMatrix.getElement(2, 3));

		Sw = SimpleOperators.multiplyMatrixProd(
				M.inverse(SimpleMatrix.InversionType.INVERT_QR), P3)
				.multipliedBy(-1.0);
		// primary angle
		double priAngle = Math.atan(Math.abs(Sw.getElement(1, 0)
				/ Sw.getElement(0, 0)));
		if (Sw.getElement(0, 0) < 0 && Sw.getElement(1, 0) >= 0)
			priAngle = Math.PI - priAngle;// in 2nd quadrant
		else if (Sw.getElement(0, 0) < 0 && Sw.getElement(1, 0) < 0)
			priAngle = Math.PI + priAngle;// in 3rd quadrant
		else if (Sw.getElement(0, 0) >= 0 && Sw.getElement(1, 0) < 0)
			priAngle = 2 * Math.PI - priAngle;// in 4th quadrant

		if (priAngle >= Math.PI)
			priAngle = -(Math.PI * 2 - priAngle);

		priAngle *= 180.0 / Math.PI;

		// secondary angle
		double sndAngle = Math.atan(Sw.getElement(2, 0)
				/ Math.sqrt(Math.pow(Sw.getElement(0, 0), 2)
						+ Math.pow(Sw.getElement(1, 0), 2)));

		sndAngle *= 180.0 / Math.PI;

		// config.getGeometry().getPrimaryAngles()[sliceNumber] = priAngle;
		// config.getGeometry().getSecondaryAngles()[sliceNumber] = sndAngle;
		// this.revalidate();
		// this.repaint();
		if (slice < errors.size()) {
			if (errors.get(slice) > backprojectionError) {
				errors.remove(slice);
				errors.add(new Double(backprojectionError));
			}
		} else {
			errors.add(new Double(backprojectionError));
		}
		return backprojectionError;
	}

}

package edu.stanford.rsl.conrad.dimreduction;


import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.jpop.OptimizableFunction;
import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.dimreduction.utils.ConfidenceMeasure;
import edu.stanford.rsl.conrad.dimreduction.utils.ConvexityTest;
import edu.stanford.rsl.conrad.dimreduction.utils.Cube;
import edu.stanford.rsl.conrad.dimreduction.utils.Error;
import edu.stanford.rsl.conrad.dimreduction.utils.FileHandler;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.Improvement;
import edu.stanford.rsl.conrad.dimreduction.utils.MyException;
import edu.stanford.rsl.conrad.dimreduction.utils.PlotIterationsError;
import edu.stanford.rsl.conrad.dimreduction.utils.PlotKError;
import edu.stanford.rsl.conrad.dimreduction.utils.PlotParameterKError;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloud;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.conrad.dimreduction.utils.ScatterPlot;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SwissRoll;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class DimensionalityReduction {
	// special parameter, you can change only here, if you use the GUI!

	// save the computed coordinate (only with the Hypercube)
	private boolean saveCoordinates = false;
	// compute with savedCoordinates (only with the Hypercube)
	private boolean withSavedCoordinates = false;

	// Optimizations....
	private Improvement im = null;

	// compute a confidence measure of the best result of all iterations and
	// save it with the filename Confidence.txt (Works only if you optimize with
	// a Lagrangian)
	private static boolean confidenceMeasure = false;

	// parameter you set in the GUI and you don't need otherwise:
	// Cube, Swiss or RealTest: (GUI)
	private String project = "";

	// For the 3Dplot PlotParameterKError and the GUI:
	// Swiss Roll:
	// Number of points:
	private int numPointsSwiss = 10;
	// Thickness number of points:
	private int thickness = 5;
	// gap width between the points:
	private double gap = 0.4;

	// Hypercube:
	// dimension:
	private int dimension = 3;
	// number of cloud points:
	private int numberOfPoints = 10;
	// edge length:
	private double edgeLength = 1.0;
	// standard deviation of the cloud points
	private double sd = 0.01;
	// additional dimensions of noise:
	private int noiseDim = 0;

	// input with GUI or direct with method:
	// RealTest;
	private String filename = "";

	// Compute a PCA:
	private static boolean computePCA = false;

	// Compute LLE:
	private static boolean computeLLE = false;

	// Parameter you can set:
	// target dimension of the optimization:
	private int targetDimension = 2;

	// Optimization function
	// 0 : Sammon objective function
	// 1 : Distance square objective function
	// 2 : Inner product objective function
	// 3 : Weighted inner product objective function
	// 4 : Lagrangian inner product objective function
	// 5 : Lagrangian distance objective function
	// 6 : Lagranian distance square objective function
	private PointCloudViewableOptimizableFunction optFunc;

	// Convexity Tests (only possible with Normal Optimization or Iterations-
	// Error - Plot
	private boolean test2D = false;
	private boolean test3D = false;
	private int dim2D = 0;
	private int dim3D = 0;
	private String convexity2Dfilename = "";
	private String convexity3Dfilename = "";

	// show original points
	private boolean showOrigPoints = false;

	// compute the SammonError
	private boolean computeError = true;
	// Standard Optimization
	private boolean normalOptimization = false;

	// Plot K - Error (save file only with GUI)
	PlotKError plotKError;

	// Plot Time - Error
	private PlotIterationsError plotIterError;

	// show best result of all iteration steps
	private boolean bestTimeValue = false;

	// Plot - parameter - k - error:
	private PlotParameterKError plotParaKError;

	// Do NOT change these parameters!!
	public static boolean runConvexityTest = false;
	private PointND[] points = null;
	private ArrayList<PointND> p = null;
	private double[][] distances = null;
	private ScatterPlot scatterPlotValues;
	private ScatterPlot scatterPlotMin;

	/**
	 * Constructor of the Dimensionality Reduction in the Point Cloud is not known yet, it has to be set later!
	 */
	public DimensionalityReduction(){
		this.targetDimension = 2;
		this.showOrigPoints = false;
		this.normalOptimization = false;
		this.test2D = false;
		this.test3D = false;
		this.bestTimeValue = false;
	}

	/**
	 * Constructor of the DimensionalityReduction
	 * @param points PointCloud of the high dimensional points
	 */
	public DimensionalityReduction(PointCloud points) {

		this.p = points.getPoints();
		this.points = HelperClass.wrapListToArray(p);
		this.distances = HelperClass.buildDistanceMatrix(HelperClass.wrapListToArray(p));
		// default settings:
		this.targetDimension = 2;
		this.showOrigPoints = false;
		this.normalOptimization = false;
		this.test2D = false;
		this.test3D = false;
		this.bestTimeValue = false;

	}

	/**
	 * Constructor of the DimensionalityReduction
	 * @param distances array of inner-point distances of the high dimensional space
	 */
	public DimensionalityReduction(double[][] distances) {
		this.distances = distances;
		// default settings:
		this.targetDimension = 2;
		this.showOrigPoints = false;
		this.normalOptimization = true;
		this.test2D = false;
		this.test3D = false;
		this.bestTimeValue = false;

	}

	/**
	 * 
	 * @return whether a confidence measure is computed
	 */
	public boolean getConfidenceMeasure() {
		return confidenceMeasure;
	}

	/**
	 * you set a confidence measure (only possible in combination with an
	 * improvement, an iterations - error plot or a standard dimensionality
	 * reduction) (the result is saved in "Confidence.txt")
	 * 
	 * @param is
	 */
	public void setConfidenceMeasure(boolean is) {
		confidenceMeasure = is;
	}

	/**
	 * 
	 * @return the PlotParameterKError
	 */
	public PlotParameterKError getPlotParameterKError() {
		return plotParaKError;
	}

	/**
	 * sets the plotParameterKError (3D plot)
	 * 
	 * @param plotParaKError
	 */
	public void setPlotParameterKError(PlotParameterKError plotParaKError) {
		this.plotParaKError = plotParaKError;
		//		plotParaKError.setDimensionalityReduction(this);
	}

	/**
	 * sets the PlotIterationsError (2D plot)
	 * 
	 * @param plotIterError
	 */
	public void setPlotIterError(PlotIterationsError plotIterError) {
		this.plotIterError = plotIterError;

	}

	/**
	 * returns weather a iterations-error-plot is computed
	 * 
	 * @return the PlotIterationsError (2D plot)
	 */
	public PlotIterationsError getPlotIterError() {	
		return this.plotIterError;
	}

	/**
	 * sets the PlotKError (2D plot)
	 * 
	 * @param plotKError
	 */
	public void setPlotKError(PlotKError plotKError) {
		this.plotKError = plotKError;

	}

	/**
	 * returns weather a k-error-plot is computed
	 * 
	 * @return the PlotKError (2D plot)
	 */
	public PlotKError getPlotKError() {
		return this.plotKError;
	}

	/**
	 * sets an Improvement (PCAoptimizer, rejectPoints or Restrictions)
	 * 
	 * @param im
	 */
	public void setImprovement(Improvement im) {
		this.im = im;
	}

	/**
	 * returns weather a improvement is used
	 * 
	 * @return the Improvement
	 */
	public Improvement getImprovement() {
		return im;
	}

	/**
	 * returns which project is used (only for parameter-k-error-plot)
	 * 
	 * @return the actual project as String
	 */
	public String getProject() {
		return project;
	}

	/**
	 * sets the project as string (Cube, Swiss, or RealData), only in GUI and
	 * PlotParameterKError
	 * 
	 * @param name
	 */
	public void setProject(String name) {
		project = name;
	}

	/**
	 * 
	 * @return the distance matrix
	 */
	public double[][] getDistances() {
		return distances;
	}
	/**
	 * sets the distance matrix
	 * 
	 * @param distances distance matrix of the high dimensional points
	 */
	public void setDistances(double[][] distances) {
		this.distances = distances; 
	}

	/**
	 * sets the points as ArrayList<PointND>
	 * 
	 * @param points ArrayList of high dimensional points
	 */
	protected void setP(ArrayList<PointND> points) {
		p = points;
	}

	/**
	 * sets the target dimension of the dimensionalityReduction
	 * 
	 * @param dim
	 *            target dimension
	 */
	public void setTargetDimension(int dim) {
		targetDimension = dim;
	}

	/**
	 * 
	 * @return the targetDimension of the DimensionalityReduction
	 */
	public int getTargetDimension() {
		return targetDimension;
	}

	/**
	 * sets the filename if real data is used (only GUI!)
	 * 
	 * @param name
	 */
	public void setFilename(String name) {
		filename = name;
	}

	/**
	 * returns the filename of the real data
	 * 
	 * @return the filename of the real data
	 */
	public String getFilename() {
		return filename;
	}

	/**
	 * sets the targetFunction of the Dimensionality Reduction, its one
	 * Lagrangian
	 * 
	 * @param optFunc
	 */
	public void setTargetFunction(PointCloudViewableOptimizableFunction optFunc) {
		this.optFunc = optFunc;
	}

	/**
	 * returns the acutal target function
	 * 
	 * @return the actual target function
	 */
	public OptimizableFunction getTargetFunction() {
		return optFunc;
	}

	/**
	 * sets the PCA optimizer
	 * 
	 * @param pca
	 */
	public static void setPCA(boolean pca) {
		computePCA = pca;
	}

	/**
	 * returns weather a PCA optimizter is used
	 * 
	 * @return whether a PCA optimizer is used
	 */
	public static boolean getPCA() {
		return computePCA;
	}

	/**
	 * sets whether a LLE will be computed
	 * 
	 * @param lle
	 */
	public static void setLLE(boolean lle) {
		computeLLE = lle;
	}

	/**
	 * returns weather a LLE is computed
	 * 
	 * @return whether a LLE is used
	 */
	public static boolean getLLE() {
		return computeLLE;
	}

	/**
	 * sets whether the Sammon Error of the chosen computation is computed
	 * 
	 * @param computeIt
	 */
	public void setComputeError(boolean computeIt) {
		computeError = computeIt;
	}

	/**
	 * returns weater the sammon error is computed
	 * 
	 * @return whether the Sammon Error will be computed
	 */
	public boolean getComputeError() {
		return computeError;
	}

	/**
	 * sets whether the Original, high-dimensional
	 * 
	 * @param showOriginalPoints
	 */
	public void setshowOrigPoints(boolean showOriginalPoints) {
		showOrigPoints = showOriginalPoints;
	}

	/**
	 * sets the standard optimization mode
	 * 
	 * @param normalOpt
	 */
	public void setNormalOptimizationMode(boolean normalOpt) {
		normalOptimization = normalOpt;
	}

	/**
	 * returns whether the standard optimization mode is used
	 * 
	 * @return whether the standard optimization mode is used
	 */
	public boolean getNormalOptimizationMode() {
		return normalOptimization;
	}

	/**
	 * sets the standard deviation of the Point Clouds of the cube (only GUI or
	 * 3Dplot!!!)
	 * 
	 * @param standarddeviation
	 */
	public void setSd(double standarddeviation) {
		sd = standarddeviation;
	}

	/**
	 * 
	 * @return the actual Standard deviation of the point clouds of the cube
	 *         (only GUI or 3Dplot!!!)
	 */
	public double getSd() {
		return sd;
	}

	/**
	 * sets the dimension of the cube (only GUI or 3Dplot!!!)
	 * 
	 * @param dim
	 */
	public void setdim(int dim) {
		dimension = dim;
	}

	/**
	 * 
	 * @return the dimension of the cube (only GUI or 3Dplot!!!)
	 */
	public int getDim() {
		return dimension;
	}

	/**
	 * sets the number of point per cloud of a cube (only GUI or 3Dplot!!!)
	 * 
	 * @param numPoints
	 */
	public void setNumPoints(int numPoints) {
		numberOfPoints = numPoints;
	}

	/**
	 * 
	 * @return the number of points per cloud of the cube (only GUI or
	 *         3Dplot!!!)
	 */
	public int getNumPoints() {
		return numberOfPoints;
	}

	/**
	 * sets the edge length of the cube (only GUI or 3Dplot!!!)
	 * 
	 * @param eL
	 */
	public void setEdgeLength(double eL) {
		edgeLength = eL;
	}

	/**
	 * 
	 * @return the edge length of the cube (only GUI or 3Dplot!!!)
	 */
	public double getEdgeLength() {
		return edgeLength;
	}

	/**
	 * sets the size of the gap of the Swiss Roll (only GUI or 3Dplot!!!)
	 * 
	 * @param newGap
	 */
	public void setGap(double newGap) {
		gap = newGap;
	}

	/**
	 * 
	 * @return the size of the gap of the Swiss Roll (only GUI or 3Dplot!!!)
	 */
	public double getGap() {
		return gap;
	}

	/**
	 * sets the thickness of the Swiss Roll (only GUI or 3Dplot!!!)
	 * 
	 * @param newThickness
	 */
	public void setThickness(int newThickness) {
		thickness = newThickness;
	}

	/**
	 * 
	 * @return the thickness of the Swiss Roll (only GUI or 3Dplot!!!)
	 */
	public int getThickness() {
		return thickness;
	}

	/**
	 * sets the number of points on a spiral of the Swiss Roll (only GUI or
	 * 3Dplot!!!)
	 * 
	 * @param numberOfPoints
	 */
	public void setNumPointsSwiss(int numberOfPoints) {
		numPointsSwiss = numberOfPoints;
	}

	/**
	 * 
	 * @return the number of points on a spiral of the Swiss Roll (only GUI or
	 *         3Dplot!!!)
	 */
	public int getNumPointsSwiss() {
		return numPointsSwiss;
	}

	/**
	 * sets whether a 2D convexity test is computed
	 * 
	 * @param convexityTest2D
	 */
	public void setConvexityTest2D(boolean convexityTest2D) {
		test2D = convexityTest2D;
	}

	/**
	 * 
	 * @return whether a 2D convexity test is computed
	 */
	public boolean getConvexityTest2D() {
		return test2D;
	}

	/**
	 * sets whether a 3D convexity test is computed
	 * 
	 * @param convexityTest3D
	 */
	public void setConvexityTest3D(boolean convexityTest3D) {
		test3D = convexityTest3D;
	}

	/**
	 * 
	 * @return whether a 3D convexity test is computed
	 */
	public boolean getConvexityTest3D() {
		return test3D;
	}

	/**
	 * sets the dimension whose convexity is proved in a 3D test
	 * 
	 * @param dim
	 */
	public void set3Ddim(int dim) {
		dim3D = dim;
	}

	/**
	 * sets the dimension whose convexity is proved in a 2D test
	 * 
	 * @param dim
	 */
	public void set2Ddim(int dim) {
		dim2D = dim;
	}

	/**
	 * sets the filename of the 2D convexity test, if the result should be saved
	 * 
	 * @param name
	 */
	public void setFilenameConvexity2D(String name) {
		convexity2Dfilename = name;
	}

	/**
	 * sets the filename of the 3D convexity test, if the result should be saved
	 * 
	 * @param name
	 */
	public void setFilenameConvexity3D(String name) {
		convexity3Dfilename = name;
	}


	/**
	 * sets whether the best result of all iteration steps is shown
	 * 
	 * @param TimeValue
	 */
	public void setBestTimeValue(boolean TimeValue) {
		bestTimeValue = TimeValue;
	}

	/**
	 * 
	 * @return whether the best result of all iteration steps will be shown
	 */
	public boolean getBestTimeValue() {
		return bestTimeValue;
	}

	/**
	 * 
	 * @return whether the coordinates of the actual hypercube optimization
	 *         should be saved
	 */
	public boolean getSaveCoordinates() {
		return saveCoordinates;
	}

	/**
	 * 
	 * @return whether saved coordinates, from a hypercube are used for the
	 *         computation
	 */
	public boolean getWithsavedCoordinates() {
		return withSavedCoordinates;
	}

	/**
	 * 
	 * @return a scatter plot (only intern use!!!)
	 */
	public ScatterPlot getScatterPlotValues() {
		return this.scatterPlotValues;
	}

	/**
	 * 
	 * @return a scatter plot (only intern use!!!)
	 */
	public ScatterPlot getScatterPlotMin() {
		return this.scatterPlotMin;
	}

	/**
	 * sets a scatter plot (only intern use!!!)
	 * 
	 * @param scatterPlotValues
	 */
	public void setScatterPlotValues(ScatterPlot scatterPlotValues) {
		this.scatterPlotValues = scatterPlotValues;
	}

	/**
	 * sets a scatter plot (only intern use!!!)
	 * 
	 * @param scatterPlotMin
	 */
	public void setScatterPlotMin(ScatterPlot scatterPlotMin) {
		this.scatterPlotMin = scatterPlotMin;
	}

	/**
	 * Runs the optimization with random initialization and displays the low dimensional points.
	 * @return an array with the low dimensional points
	 */
	public double [] optimize() {
		return optimize(true, System.currentTimeMillis());
	}

	/**
	 * Main Function of the Project, starts the optimization
	 * @param showPoints shows a 3D visualization of the low dimensional points, if true
	 * @param seed allows to set the random seed for reproducible results.
	 * @return an array with the low dimensional points
	 */
	public double [] optimize(boolean showPoints, long seed) {

		double[] result = null;

		if (showOrigPoints && p != null) {
			System.out.println("Show original points");
			PointCloudViewer pcvOrig = new PointCloudViewer("Orig", p);
			pcvOrig.setVisible(true);
		}

		if (normalOptimization || plotIterError != null || im != null) {
			System.out.println("optimization chosen!");
			if (im != null) {

				try {
					double[] temp = im.improve(this, distances);
					if (temp != null) {
						result = temp;
						this.distances = im.getDistances();
					}
				} catch (Exception e) {
					e.printStackTrace();
				}

			}
			double[] initial = null;
			if (im == null || im.getRestriction()) {
				System.out.println("Lagrangian Optimization build");
				LagrangianOptimization laop = new LagrangianOptimization(
						distances, targetDimension, this, optFunc);

				// initialization:
				if (distances != null) {
					initial = laop.randomInitialization(seed);
					System.out.println("Random initialization found.");
				}

				try {
					if (distances != null) {
						System.out.println("Optimize...");
						result = laop.optimize(initial, showPoints);
						if (this.getConvexityTest2D()
								|| this.getConvexityTest3D()) {
							this.convexityTest(optFunc, initial);
						}
						System.out.println("Optimization finished.");
						if (this.getConvexityTest2D()
								|| this.getConvexityTest3D()) {
							this.convexityTest(optFunc, result);
						}
						if (computeError) {
							System.out.println("Compute Error");
							Error error = new Error();
							System.out.println(error.computeError(HelperClass
									.wrapArrayToPoints(result,
											distances.length), "Result",
									distances));
						}
					}
					if (plotIterError != null) {
						System.out
						.println("Compute number of iterations - error - plot (2D)");
						try {
							plotIterError.computePlot_time_error(this, laop);
						} catch (IOException e) {
							e.printStackTrace();
						}
					}

					if (bestTimeValue) {

						double[] bestResult = this
								.getBestResultOfAllIterations(laop);
						HelperClass.showPoints(bestResult, bestResult.length
								/ targetDimension,
								"best result of all iteration steps");
					}

				} catch (Exception e) {
					e.printStackTrace();
				}
			}

		}
		// mean value - error plot:
		// PlotHelper.plot2D(PointCloudViewableFunction.getDistMean(),
		// PointCloudViewableFunction.getErrors());

		if (plotKError != null) {
			System.out.println("Compute k - error - plot (2D)");
			try {
				plotKError.computePlot_k_error(distances, this);
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		if (this.plotParaKError != null) {

			try {
				this.plotParaKError.setDimensionalityReduction(this); 
				this.plotParaKError.computePlot_parameter_k_error();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		if (confidenceMeasure) {

			if (result == null || distances == null) {
				System.err
				.println("You have to compute a normal optimization, a time - error plot or use a improvement to compute a confidence measure!");
			} else {
				System.out.println("Compute a confidence measure.");
				double[][] reconstructedDistances = HelperClass
						.buildDistanceMatrix(HelperClass
								.wrapListToArray(HelperClass.wrapArrayToPoints(
										result, distances.length)));
				try {
					new ConfidenceMeasure(distances, reconstructedDistances);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		if (computePCA) {
			System.out.println("Compute a PCA");
			try {
				computePCA(computeError, points);
			} catch (MyException e) {
				e.printStackTrace();
			}
		}

		if (computeLLE) {
			System.out.println("Compute a LLE");
			try {
				computeLLE(computeError, points);
			} catch (MyException e) {
				e.printStackTrace();
			}
		}
		return result;
	}

	/**
	 * 
	 * @param laop
	 *            actual Lagrangian optimization
	 * @return the best result of all iteration steps according to the smallest
	 *         Sammon Error
	 */
	public double[] getBestResultOfAllIterations(LagrangianOptimization laop) {

		System.out.println("Smallest error of all iteration steps : "
				+ laop.getMinimumError());
		return laop.getBestTimeValue();

	}

	/**
	 * computes the coordinates of the actual project, by using the following
	 * parameters
	 * 
	 * @param numPointsSwiss
	 * @param thickness
	 * @param gap
	 * @param dimension
	 * @param numberOfPoints
	 * @param edgeLength
	 * @param sd
	 * @param saveCoordinates
	 * @param withsavedCoordinates
	 * @throws Exception
	 */
	public void computeCoordinates(int numPointsSwiss, int thickness,
			double gap, int dimension, int numberOfPoints, double edgeLength,
			double sd, boolean saveCoordinates, boolean withsavedCoordinates)
					throws Exception {
		if (project.length() == 0 && p != null) {

			points = HelperClass.wrapListToArray(p);
			distances = HelperClass.buildDistanceMatrix(points);
		} else if (project.length() == 0 && p == null) {
			// nothing to do
		} else if (project == "SwissRoll" || project == "Cube") {
			if (project == "SwissRoll") {
				SwissRoll sr = new SwissRoll(gap, numPointsSwiss, thickness);
				ArrayList<PointND> Swiss = sr.getPointList();
				PointND[] pointsSwiss = sr.getPoints();
				p = Swiss;
				points = pointsSwiss;
			} else if (project == "Cube") {
				Cube c = new Cube(edgeLength, dimension,
						sd, numberOfPoints, withsavedCoordinates,
						saveCoordinates, noiseDim);
				ArrayList<PointND> cube = c.getPointList();
				PointND[] pointsCube = c.getPoints();
				p = cube;
				points = pointsCube;
			}

			distances = HelperClass.buildDistanceMatrix(points);

		} else if (project == "RealTest") {
			try {
				distances = FileHandler.loadData(filename);

			} catch (MyException e) {
				System.err.println(e.getMessage());
				System.exit(0);
			}
		} else {
			throw new MyException("This is no available testcase!!");
		}

	}

	/**
	 * calls the class LLE to compute a LLE
	 * 
	 * @param numNeighbors
	 *            number of Neighbors
	 * @param computeError
	 *            boolean, whether the error should be computed
	 * @throws MyException
	 */
	private void computeLLE(boolean computeError, PointND[] points)
			throws MyException {

		LocallyLinearEmbedding locLinEmb = new LocallyLinearEmbedding(points);
		locLinEmb.setNumNeighbors(12);
		locLinEmb.computeLLE();
		ArrayList<PointND> lle = locLinEmb.getPoints();
		PointCloudViewer pcvLLE = new PointCloudViewer("LLE", lle);
		pcvLLE.setVisible(true);
		Error errorLLE = new Error();
		System.out.println("Error of LLE: "
				+ errorLLE.computeError(lle, "LLE", distances));

	}

	/**
	 * calls the class PCA to compute a PCA
	 * 
	 * @param dimPCA
	 *            target dimension
	 * @param computeError
	 *            boolean, whether the error should be computed
	 * @param points
	 *            actual points
	 * @throws MyException
	 */
	private void computePCA(boolean computeError, PointND[] points)
			throws MyException {
		PCA pca = new PCA();
		pca.setTargetDimension(2);
		ArrayList<PointND> PCVectors = pca.computePCA(points);
		PointCloudViewer pcvPCA = new PointCloudViewer("PCA", PCVectors);
		pcvPCA.setVisible(true);
		Error errorPCA = new Error();
		System.out.println("Error of PCA: "
				+ errorPCA.computeError(PCVectors, "PCA", distances));

	}

	/**
	 * organizes the convexity tests with the points dim2D and dim3D
	 * 
	 * @param function
	 *            actual OptimizableFunction
	 * @param initial
	 *            actual coordinates of the points
	 * @throws Exception
	 *             if there are less points than the index dim2D or dim3D
	 */
	void convexityTest(OptimizableFunction function, double[] initial)
			throws Exception {
		if (dim2D > distances.length) {
			throw new MyException(
					"Dimension of the 2D-convexity test is too big!!");
		}
		if (dim3D > distances.length) {
			throw new MyException(
					"Dimension of the 3D-convexity test is too big!!");

		} else {

			if (test2D) {
				ConvexityTest.convexityTest2D(function, initial, dim2D,
						distances.length, convexity2Dfilename, distances, targetDimension);
			}
			if (test3D) {
				ConvexityTest.convexityTest3D(function, initial, dim3D,
						distances.length, convexity3Dfilename, distances, targetDimension);
			}
		}
	}

}
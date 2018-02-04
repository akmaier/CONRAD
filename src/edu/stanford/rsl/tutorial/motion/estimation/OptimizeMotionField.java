package edu.stanford.rsl.tutorial.motion.estimation;

import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ThinPlateSplinesBasedProjectionWarpingTool;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceUniformCubicBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.motion.compensation.OpenCLCompensatedBackProjectorTPS;

public class OptimizeMotionField {

	/**
	 * Initial motionfield. Computed in InitialOptimization
	 */
	private float[][] motionfield;

	/**
	 * Contains filtered projection images
	 * Needs to be initialized and loaded outside of this class
	 */
	private ProjectionLoader pLoad;

	/**
	 * 2-D positions of diaphragm vertices on detector
	 */
	private double[][] diaCoords;

	/**
	 * Spline of the diaphragm
	 */
	private SurfaceBSpline dia;
	/**
	 * 3-D Position of the diaphragm top measured by diaphragmtrackingtool
	 */
	private double[] diaPositionField;
	/**
	 * the spline projections will be rendered in this grid
	 */
	private Grid3D diaRendered;
	/**
	 * parameters of the motionfield, as estimated by InitialMotionfield
	 */
	private float[] params;
	/**
	 * parabola parameters as estimated by diaphragmtrackingtool
	 */
	private double[][] diaModel;

	private float[] motion;
	private int compressionLowerBorder;
	private int diaBorder;

	public OptimizeMotionField(float[][] motionfield, float[] params, SurfaceBSpline dia, ProjectionLoader pLoad) {
		Configuration.loadConfiguration();
		this.motionfield = motionfield;
		this.params = params;
		this.pLoad = pLoad;
		this.dia = dia;
		this.diaPositionField = Configuration.getGlobalConfiguration().getDiaphragmPositionField();
		this.diaCoords = Configuration.getGlobalConfiguration().getDiaphragmCoords();
		this.diaModel = Configuration.getGlobalConfiguration().getDiaphragmModelField();

		double oZ = Configuration.getGlobalConfiguration().getGeometry().getOriginZ();
		double spacingZ = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();

		compressionLowerBorder = (int) ((params[4])
				* Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ() / 10.f * spacingZ + oZ);
		diaBorder = (int) params[5] + InitialOptimization.diaOffsetMM;
	}

	private TimeVariantSurfaceBSpline optimize() throws Exception {
		double oZ = Configuration.getGlobalConfiguration().getGeometry().getOriginZ();

		double spacingZ = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();
		int diaBorderPX = (int) ((params[5] - oZ) / spacingZ) + (int) (InitialOptimization.diaOffsetMM / spacingZ);

		motion = new float[motionfield.length];
		for (int i = 0; i < motion.length; i++) {
			motion[i] = motionfield[i][diaBorderPX];
		}

		OpenCLSplineRenderer splineRender = new OpenCLSplineRenderer();

		//		splineRender.renderAppendBufferToGrid(dia);

		TimeVariantSurfaceBSpline tSpline = createTimeVariantSpline(dia);
		diaRendered = splineRender.SurfaceBSplineRenderingAppendBufferToGrid(tSpline);

		//for each projection
		int width = 20;
		double evalTop[] = new double[motionfield.length];
		double evalFull[] = new double[motionfield.length];
		VisualizationUtil.createPlot(motion).show();
		for (int i = 0; i < motionfield.length; i++) {

			int[] diaSplineContour = getDiaphragmContour(i);
			int diaTopX = getDiaphragmTop(diaSplineContour);

			evalTop[i] = compare(diaSplineContour, i, diaTopX - width, diaTopX + width);
			if (Math.abs(evalTop[i]) > 1.0) {
				SimpleVector splinePt3D = triangulationTrajectory(i,
						new SimpleVector(diaTopX, diaSplineContour[diaTopX], 1.0));
				SimpleVector diaOrig = new SimpleVector(diaPositionField[i * 3], diaPositionField[i * 3 + 1],
						diaPositionField[i * 3 + 2]);
				splinePt3D.subtract(diaOrig);
				motion[i] += splinePt3D.getElement(2);
			}

		}
		tSpline = createTimeVariantSpline(dia);
		diaRendered = splineRender.SurfaceBSplineRenderingAppendBufferToGrid(tSpline);
		for (int i = 0; i < motionfield.length; i++) {
			int[] diaSplineContour = getDiaphragmContour(i);
			int[] minMaxDia = minMaxContour(diaSplineContour);
			evalFull[i] = compare(diaSplineContour, i, minMaxDia[0], minMaxDia[1]);
		}
		VisualizationUtil.createPlot(evalTop).show();
		VisualizationUtil.createPlot(evalFull).show();
		VisualizationUtil.createPlot(motion).show();

		double avgEvalFull = 0;
		double absError = 0;
		for (int i = 0; i < evalTop.length; i++) {
			avgEvalFull += evalFull[i];
			absError += Math.abs(evalFull[i]);
		}
		absError /= evalTop.length;
		avgEvalFull /= evalTop.length;

		double sign = Math.signum(avgEvalFull);

		int bestCompression = (int) (params[4]
				* Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ() / 10.f * spacingZ + oZ);
		for (int j = 1; j < 10; j++) {
			compressionLowerBorder = (int) ((int) (int) (params[4]
					* Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ() / 10.f * spacingZ + oZ)
					+ Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ() / 20.f * spacingZ
							* (-sign * j));
			tSpline = createTimeVariantSpline(dia);
			diaRendered = splineRender.SurfaceBSplineRenderingAppendBufferToGrid(tSpline);
			double eval[] = new double[motionfield.length];
			for (int i = 0; i < motionfield.length; i++) {

				int[] diaSplineContour = getDiaphragmContour(i);

				int[] minMaxDia = minMaxContour(diaSplineContour);
				eval[i] = compare(diaSplineContour, i, minMaxDia[0], minMaxDia[1]);

			}
			VisualizationUtil.createPlot(eval).show();
			double avgEval = 0;
			for (int k = 0; k < eval.length; k++) {
				avgEval += Math.abs(eval[k]);
			}
			avgEval /= eval.length;
			System.out.println(avgEval);
			System.out.println(absError);
			if (absError < avgEval) {
				break;
			} else {
				bestCompression = (int) ((int) (int) (params[4]
						* Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ() / 10.f * spacingZ
						+ oZ)
						+ Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ() / 20.f * spacingZ
								* (-sign * j));
				absError = avgEval;

			}
		}

		System.out.println(bestCompression);
		compressionLowerBorder = bestCompression;

		tSpline = createTimeVariantSpline(dia);

		return tSpline;
	}

	public OpenCLCompensatedBackProjectorTPS optimalReconstructor() throws Exception {
		TimeVariantSurfaceBSpline tSpline = optimize();

		ArrayList<PointND> pointsZero = new ArrayList<PointND>();
		ArrayList<PointND> valsZero = new ArrayList<PointND>();

		int zPlane = Integer
				.parseInt(JOptionPane.showInputDialog("Enter z-position of top of the heart in world-coords: ", 0));
		int oX = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginX();

		int oY = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginY();

		int stepSizeX = -oX / 4;
		int stepSizeY = -oY / 4;
		PointND np = new PointND(0, 0, 0);
		for (int i = oX; i < -oX; i += stepSizeX) {
			for (int j = oY; j < -oY; j += stepSizeY) {
				PointND zp = new PointND(i, j, zPlane);
				pointsZero.add(zp);
				valsZero.add(np);
			}
		}

		float[][] A = new float[motion.length][];
		float[][] b = new float[motion.length][];
		float[][] coeff = new float[motion.length][];
		float[][] pts = new float[motion.length][];

		int uCtrl = tSpline.getSplines().get(0).getNumberOfUPoints();
		int vCtrl = tSpline.getSplines().get(0).getNumberOfVPoints();
		int stepU = uCtrl / 4;
		int stepV = vCtrl / 4;
		int[] sampleList = new int[5 * 5];
		int sampleI = 0;
		for (int w = 0; w < vCtrl; w += stepV) {
			for (int z = 0; z < uCtrl; z += stepU) {
				sampleList[sampleI++] = w * uCtrl + z;
			}
		}

		ArrayList<PointND> p0 = tSpline.getControlPoints(0);
		for (int i = 0; i < motion.length; i++) {
			ArrayList<PointND> tempPoints = tSpline.getControlPoints(i);
			ArrayList<PointND> points = new ArrayList<PointND>();
			for (int z = 0; z < sampleList.length; z++) {
				points.add(tempPoints.get(sampleList[z]).clone());
			}
			points.addAll(pointsZero);
			ArrayList<PointND> values = new ArrayList<PointND>();

			for (int j = 0; j < sampleList.length; j++) {
				SimpleVector p = p0.get(sampleList[j]).getAbstractVector().clone();
				p.subtract(points.get(j).getAbstractVector().clone());
				values.add(new PointND(p));

			}

			values.addAll(valsZero);
			ThinPlateSplineInterpolation tpsI = new ThinPlateSplineInterpolation(3, points, values);
			A[i] = tpsI.getAsFloatA();
			b[i] = tpsI.getAsFloatB();
			coeff[i] = tpsI.getAsFloatCoeffs();
			pts[i] = tpsI.getAsFloatPoints();
			System.out.println("TPS-Interpolation:" + i);
			/*Grid3D grid = new Grid3D(256,256,256);
			if(i==80){
			for (int q = 0; q < 256; q++) {
				for (int w = 0; w < 256; w++) {
					for (int e = 0; e < 256; e++) {
			
						float xi = 0;
						float yi = 0;
						float zi = 0;
						float x1 = q-128;
						float y1 = w-128;
						float z1 = e-128;
						int ptsNr = pts[i].length/3;
						for(int l = 0; l < ptsNr; l++) {
						
							float x2 = pts[i][l*3];
							float y2 = pts[i][l*3+1];
							float z2 = pts[i][l*3+2];
							float sum = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
							float dist =  (float) Math.sqrt(sum);
							xi = xi+ coeff[i][l*3]*dist;		
							yi = yi+ coeff[i][l*3+1]*dist;
							zi = zi+ coeff[i][l*3+2]*dist;
			
						}
					
						for(int l = 0 ; l < 3; l++) {
						
						
							xi = xi+ A[i][l*3]*x1;
							yi = yi+ A[i][l*3+1]*y1;
							zi = zi+ A[i][l*3+2]*z1;
							
						}
						xi = xi+ b[i][0];
						yi = yi+b[i][1];
						zi = zi+b[i][2];
			
						grid.setAtIndex(q,w,e, zi);
						
					//grid.setAtIndex(q,w,e, (float) tpsI
					//		.interpolate(new PointND(q-128,w-128,e-128)).getElement(2));
				}
				}
			}
			grid.show();
			}*/
		}
		OpenCLCompensatedBackProjectorTPS otps = new OpenCLCompensatedBackProjectorTPS(coeff, pts, A, b);
		return otps;
	}

	private int[] minMaxContour(int[] diaContour) {
		int[] minmax = new int[2];

		int i = 0;
		while (true) {
			if (diaContour[i] != 0) {
				break;
			}
			i++;
		}
		minmax[0] = i;
		i = diaContour.length - 1;
		while (true) {
			if (diaContour[i] != 0) {
				break;
			}
			i--;
		}
		minmax[1] = i;
		return minmax;
	}

	private TimeVariantSurfaceBSpline createTimeVariantSpline(SurfaceBSpline spline) throws IOException {
		ArrayList<SurfaceBSpline> splines = new ArrayList<SurfaceBSpline>();

		ArrayList<PointND> cPoints = spline.getControlPoints();

		for (int i = 0; i < motionfield.length; i++) {
			ArrayList<PointND> newPoints = new ArrayList<PointND>();
			for (PointND p : cPoints) {
				newPoints.add(computeCompression(p, i));
			}
			splines.add(new SurfaceUniformCubicBSpline(newPoints, spline.getUKnots(), spline.getVKnots()));
		}

		TimeVariantSurfaceBSpline tSpline = new TimeVariantSurfaceBSpline(splines);
		return tSpline;
	}

	/**
	 * Compresses a point with the signal measured at top of diaphragm
	 * @param lowerBorder lower border of diaphragm
	 * @param diaphragmTop diaphragm vertex
	 * @param point the point at which the compression is computed
	 * @param motion motionfield at top
	 * @param i projection number
	 * @return compressed point
	 */
	private PointND computeCompression(PointND point, int i) {

		int min = compressionLowerBorder;
		int max = diaBorder;
		double dist = max - min;
		PointND p = point.clone();
		int z = (int) point.get(2);

		if (z >= min && z <= max) {
			p.set(2, (float) (p.get(2) + motion[i] * (dist - (max - z)) / dist));
			return p;
		} else if (z < min) {
			return p;
		} else if (z > max) {
			p.set(2, (float) (p.get(2) + motion[i]));
			return p;
		}

		return p;
	}

	/**
	 * Searches for the maximum of the diaphragm contour
	 * @param diaContour array that contains the y-coordinates at each x
	 * @return maximum of contour
	 */
	private int getDiaphragmTop(int[] diaContour) {
		int tmp = Integer.MAX_VALUE;
		int index = 0;
		int cnt = 0;
		for (int i = 0; i < diaContour.length; i++) {

			if (diaContour[i] < tmp && diaContour[i] > 0) {
				tmp = diaContour[i];
				cnt = 0;
				while (true) {
					if (diaContour[i] != tmp) {
						i--;
						break;
					}
					i++;
					cnt++;

				}
				index = i - cnt / 2;
			}
		}
		return index;
	}

	/**
	 * Finds the diaphragm contour of the rendered diaphragm spline by searching for each x-coordinate the largest non-zero element
	 * @param projectionNumber projection index
	 * @return array that contains the y-indices of the diaphragm contour
	 */
	private int[] getDiaphragmContour(int projectionNumber) {
		int[] coords = new int[diaRendered.getSize()[0]];

		for (int i = 0; i < diaRendered.getSize()[0]; i++) {
			for (int j = 0; j < diaRendered.getSize()[1]; j++) {

				if (diaRendered.getAtIndex(i, j, projectionNumber) != 0) {
					coords[i] = j;
					break;
				}
			}
		}

		return coords;
	}

	/**
	 * Compares projection of the volume with an original projection
	 * @param proj projection image
	 * @param original original projection image
	 * @param start start x-coord of comparison
	 * @param end end x-coord of comparison
	 * @return positive value when spline needs to be moved up, negative when moved down
	 */
	private double compare(int[] projCoordsY, int original, int start, int end) {

		double a = diaModel[original][0];
		double b = diaModel[original][1];
		double c = diaModel[original][2];
		double roiWidthLeft = diaModel[original][4];
		double roiHeightTop = diaModel[original][3];
		double ptX = diaCoords[original][0];
		double ptY = diaCoords[original][1];

		Grid2D projectionOrig = pLoad.getProjections().get(original);

		int maxX = projectionOrig.getWidth();

		if (start < 0)
			start = 0;
		if (end >= maxX)
			end = maxX - 1;

		int range = end - start + 1;

		double eval = 0;

		for (int xI = start; xI < end; xI++) {

			int x = (int) (xI - ptX + roiWidthLeft);
			int y = (int) (a * x * x + b * x + c);
			int yI = (int) (y + ptY - roiHeightTop);

			int yM = projCoordsY[xI];

			eval += (yI - yM);

		}

		return eval / range;
	}

	/**
	 * Computes cross product of two vectors
	 * @param v1 vector 1 
	 * @param v2 vector 2
	 * @return crossproduct
	 */
	private SimpleVector crossProduct(SimpleVector v1, SimpleVector v2) {
		SimpleVector sol = new SimpleVector(v1.getLen());

		sol.setElementValue(0, v1.getElement(1) * v2.getElement(2) - v1.getElement(2) * v2.getElement(1));
		sol.setElementValue(1, v1.getElement(2) * v2.getElement(0) - v1.getElement(0) * v2.getElement(2));
		sol.setElementValue(2, v1.getElement(0) * v2.getElement(1) - v1.getElement(1) * v2.getElement(0));

		return sol;
	}

	private SimpleVector triangulationTrajectory(int projectionNumber, SimpleVector detectorCoords) {

		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();

		SimpleVector pt3D = new SimpleVector(diaPositionField[projectionNumber * 3],
				diaPositionField[projectionNumber * 3 + 1], diaPositionField[projectionNumber * 3 + 2]);

		SimpleVector n = new SimpleVector(new double[] { 0, 0, 1 });
		SimpleVector q = geom.getProjectionMatrix(projectionNumber).computeCameraCenter();
		SimpleVector v = geom.getProjectionMatrix(projectionNumber).computeRayDirection(detectorCoords);
		SimpleVector w = crossProduct(n, v);
		SimpleVector z = q.clone();
		q.subtract(pt3D);

		SimpleVector b = crossProduct(q, n);
		double factor2 = SimpleOperators.multiplyInnerProd(b, w) / SimpleOperators.multiplyInnerProd(w, w);
		z.add(v.multipliedBy(factor2));

		SimpleVector sol = new SimpleVector(3);
		sol.setSubVecValue(0, pt3D);
		sol.setElementValue(2, z.getElement(2));

		return sol;
	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
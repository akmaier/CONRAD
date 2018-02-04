package edu.stanford.rsl.tutorial.motion.estimation;

import ij.ImageJ;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceUniformCubicBSpline;
import edu.stanford.rsl.conrad.io.VTKMeshReader;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;

/**
 * This class estimates a cubic B-Spline Surface given a set of points
 * 
 * @author Marco Boegel
 * 
 */
public class EstimateBSplineSurface {

	/**
	 * Input points to which the surface is to be fitted
	 */
	private ArrayList<PointND> gridPoints;

	/**
	 * Parameter vectors in u and v-direction
	 */

	private double[] uPara, vPara;

	/**
	 * Number of controlpoints in u-direction the spline is supposed to have
	 */
	private int uCtrl = 21;

	/**
	 * Number of controlpoints in v-direction the spline is supposed to have
	 */
	private int vCtrl = 21;

	/**
	 * Getter for u-controlpoints
	 * 
	 * @return number of u-controlpoints
	 */
	public int getuCtrl() {
		return uCtrl;
	}

	/**
	 * Setter for number of u-controlpoints
	 * 
	 * @param uCtrl
	 *            number of controlpoints in u-direction
	 */
	public void setuCtrl(int uCtrl) {
		this.uCtrl = uCtrl;
	}

	/**
	 * Getter for number of v-controlpoints
	 * 
	 * @return number of controlpoints in v-direction
	 */
	public int getvCtrl() {
		return vCtrl;
	}

	/**
	 * Setter for number of v-controlpoints
	 * 
	 * @param vCtrl
	 *            number of controlpoints in v-direction
	 */
	public void setvCtrl(int vCtrl) {
		this.vCtrl = vCtrl;
	}

	/**
	 * Constructor
	 * 
	 * @param gridPoints
	 *            points to be spline-fitted
	 */
	public EstimateBSplineSurface(ArrayList<PointND> gridPoints) {

		this.gridPoints = gridPoints;
	}

	/**
	 * Initializes the gridPoints
	 * 
	 * @param sampling
	 *            sampling stepsize
	 * @return Lists of samplepoints ( inner list are u parameterized)
	 */
	private ArrayList<ArrayList<PointND>> init(int sampling) {
		assert (sampling > 0);

		ArrayList<ArrayList<PointND>> output = new ArrayList<ArrayList<PointND>>();

		PointND max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		PointND min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		for (PointND p : gridPoints) {
			max.updateIfHigher(p);
			min.updateIfLower(p);
		}

		Configuration.loadConfiguration();
		Configuration c = Configuration.getGlobalConfiguration();
		Grid3D grid = new Grid3D(c.getGeometry().getReconDimensionX(), c.getGeometry().getReconDimensionY(),
				c.getGeometry().getReconDimensionZ());
		double oX = c.getGeometry().getOriginX();
		double oY = c.getGeometry().getOriginY();
		double oZ = c.getGeometry().getOriginZ();
		double vX = c.getGeometry().getVoxelSpacingX();
		double vY = c.getGeometry().getVoxelSpacingY();
		double vZ = c.getGeometry().getVoxelSpacingZ();
		// Data In NIFTI Format. x and y have reversed signs.
		for (int i = 0; i < gridPoints.size(); i++) {
			grid.setAtIndex(-(int) Math.round((gridPoints.get(i).get(0) + oX) / vX),
					-(int) Math.round((gridPoints.get(i).get(1) + oY) / vY),
					(int) (Math.round(gridPoints.get(i).get(2) - oZ) / vZ), 10);
		}
		int maxX = -(int) Math.round((min.get(0) + oX) / vX);
		int maxY = -(int) Math.round((min.get(1) + oY) / vY);
		int minX = -(int) Math.round((max.get(0) + oX) / vX);
		int minY = -(int) Math.round((max.get(1) + oY) / vY);

		double distU = maxX - minX;
		double distV = maxY - minY;

		uPara = computeParamUniform((int) distU / sampling + 1);
		vPara = computeParamUniform((int) distV / sampling + 1);

		// We sample the points by picking the largest z-Value at each point
		// (x,y)
		// We save these points in a 2D Grid
		Grid2D samples = new Grid2D(grid.getSize()[0], grid.getSize()[1]);

		for (int i = 0; i < grid.getSize()[0]; i++) {
			for (int j = 0; j < grid.getSize()[1]; j++) {
				int k = grid.getSize()[2] - 1;
				double val = grid.getAtIndex(i, j, k);
				// valid gridpoints have val==10
				while (val != 10 && k > 0) {
					k--;
					val = grid.getAtIndex(i, j, k);
				}
				if (k == 0)
					samples.setAtIndex(i, j, grid.getSize()[2] - 1);
				else
					samples.setAtIndex(i, j, k);
			}
		}
		grid.show();
		// sample in y-direction
		for (int j = minY; j <= maxY; j += sampling) {
			ArrayList<PointND> list = new ArrayList<PointND>();
			int x0 = minX;
			int x0Max = 1;
			int x1Max = 0;
			// select the largest distance in x-direction with valid values
			// this removes zig-zag outliers from the segmentation
			while (x0 < maxX) {
				double val = samples.getAtIndex(x0, j);
				while (val == grid.getSize()[2] - 1 && x0 < maxX) {
					x0++;
					val = samples.getAtIndex(x0, j);
				}
				if (x0 == maxX)
					break;

				int x1 = x0;
				val = samples.getAtIndex(x1, j);
				while (val != grid.getSize()[2] - 1 && x1 < maxX) {
					x1++;
					val = samples.getAtIndex(x1, j);
				}
				x1--;
				if ((x1 - x0) > (x1Max - x0Max)) {
					x1Max = x1;
					x0Max = x0;
				}
				x0 = x1 + 1;
			}

			double sampleDist = (x1Max - x0Max) / (double) (uPara.length - 1);

			// sample points, u first
			for (int idx = 0; idx < uPara.length; idx++) {
				double i = x0Max + idx * sampleDist;
				PointND p = new PointND(i * vX + oX, j * vY + oY, samples.getAtIndex((int) Math.round(i), j) * vZ + oZ);
				list.add(p);
			}
			output.add(list);
		}

		return output;
	}

	/**
	 * Computes uniform parameter vector
	 * 
	 * @param length
	 *            length of the vector
	 * @return array of length, values [0,1]
	 */
	private static double[] computeParamUniform(int length) {
		int N = length - 1;
		double delta = 1.0 / N;
		double[] knots = new double[length];
		for (int i = 0; i < length; i++) {
			knots[i] = i * delta;
		}

		return knots;
	}

	/**
	 * This method provides the interface to start the surface fitting
	 * 
	 * @param sampling
	 *            sampling stepsize of the provided gridPoints
	 * @return uniform Cubic B-Spline Surface
	 */
	public SurfaceUniformCubicBSpline estimateUniformCubic(int sampling) {
		ArrayList<ArrayList<PointND>> measurements = init(sampling);

		return estimateUniformCubicInternal(measurements);
	}

	/**
	 * This method does the internal B-Spline fitting. It calls the 2D BSpline
	 * estimator (EstimateCubicBSpline2D), to fit splines in u-direction and
	 * subsequently interpolates these splines in v-direction implemented based
	 * on http
	 * ://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/SURF-APP-global.
	 * html
	 * 
	 * @param points
	 *            sampled points
	 * @return uniform Cubic B-Spline Surface
	 */
	private SurfaceUniformCubicBSpline estimateUniformCubicInternal(ArrayList<ArrayList<PointND>> points) {
		// knots = ctrl +order +1

		int vL = vPara.length;

		// init to remove error
		double[] uKnots = new double[1];
		double[] vKnots = new double[1];

		// intermediate controlpoints of the u-Splines
		ArrayList<ArrayList<PointND>> cList = new ArrayList<ArrayList<PointND>>();

		// fit u-Splines first
		for (int d = 0; d < vL; d++) {

			EstimateCubic2DSpline estimator = new EstimateCubic2DSpline(points.get(d));
			ArrayList<PointND> list = estimator.estimateUniformCubic(uCtrl).getControlPoints();

			cList.add(list);

			if (d == vL - 1)
				uKnots = estimator.getKnots();

		}

		// Controlpoints of the Surface Spline
		ArrayList<PointND> ControlPoints = new ArrayList<PointND>();

		// interpolate the u-Splines to get the surface
		for (int c = 0; c < uCtrl; c++) {
			ArrayList<PointND> gPoints = new ArrayList<PointND>();
			for (int i = 0; i < vL; i++) {
				gPoints.add(cList.get(i).get(c));
			}

			EstimateCubic2DSpline estimator = new EstimateCubic2DSpline(gPoints);

			ControlPoints.addAll(estimator.estimateUniformCubic(vCtrl).getControlPoints());

			if (c == uCtrl - 1)
				vKnots = estimator.getKnots();
		}

		return new SurfaceUniformCubicBSpline(ControlPoints, uKnots, vKnots);

	}

	public static void main(String args[]) throws Exception {
		int sampling = 1;
		VTKMeshReader vRead = new VTKMeshReader();
		new ImageJ();

		String filename = FileUtil.myFileChoose(".vtk", false);
		vRead.readFile(filename);
		EstimateBSplineSurface estimator = new EstimateBSplineSurface(vRead.getPts());
		SurfaceUniformCubicBSpline spline = estimator.estimateUniformCubic(sampling);

		Configuration.loadConfiguration();
		Configuration c = Configuration.getGlobalConfiguration();
		Grid3D grid = new Grid3D(c.getGeometry().getReconDimensionX(), c.getGeometry().getReconDimensionY(),
				c.getGeometry().getReconDimensionZ());

		for (int i = 0; i < grid.getSize()[0]; i++) {
			double u = ((double) i) / (grid.getSize()[0]);
			for (int j = 0; j < grid.getSize()[1]; j++) {
				double v = ((double) j) / (grid.getSize()[1]);
				PointND p = spline.evaluate(u, v);

				if (0 <= -((p.get(0) + c.getGeometry().getOriginX()) / c.getGeometry().getVoxelSpacingX())
						&& 0 <= -((p.get(1) + c.getGeometry().getOriginY()) / c.getGeometry().getVoxelSpacingY())
						&& 0 <= ((p.get(2) - c.getGeometry().getOriginZ()) / c.getGeometry().getVoxelSpacingZ())
						&& -((p.get(0) + c.getGeometry().getOriginX()) / c.getGeometry().getVoxelSpacingX()) < grid
								.getSize()[0]
						&& -((p.get(1) + c.getGeometry().getOriginY()) / c.getGeometry().getVoxelSpacingY()) < grid
								.getSize()[1]
						&& ((p.get(2) - c.getGeometry().getOriginZ()) / c.getGeometry().getVoxelSpacingZ()) < grid
								.getSize()[2])
					grid.setAtIndex(
							(int) -((p.get(0) + c.getGeometry().getOriginX()) / c.getGeometry().getVoxelSpacingX()),
							(int) -((p.get(1) + c.getGeometry().getOriginY()) / c.getGeometry().getVoxelSpacingY()),
							(int) ((p.get(2) - c.getGeometry().getOriginZ()) / c.getGeometry().getVoxelSpacingZ()), 10);

			}
		}

		grid.show();

	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
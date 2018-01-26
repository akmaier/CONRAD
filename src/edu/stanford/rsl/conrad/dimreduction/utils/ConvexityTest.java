package edu.stanford.rsl.conrad.dimreduction.utils;


import java.io.IOException;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.jpop.OptimizableFunction;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class ConvexityTest {

	/**
	 * Plots a 2D plot where the first coordinate of the "dim" point is moved,
	 * you can see whether the function is convex in this area.
	 * 
	 * @param function
	 *            the optimization function that is used
	 * @param initial
	 *            the actual coordinates
	 * @param dim
	 *            the number of the point which will be moved
	 * @param maxDim
	 *            the number of points
	 * @param filename
	 *            the filename, if the result is going to be saved
	 * @param distances
	 *            matrix of inner-point distances in the original space
	 * @param targetDimension
	 *            target dimension of the mapping
	 * @throws Exception
	 */
	public static void convexityTest2D(OptimizableFunction function, double[] initial, int dim, int maxDim, String filename,
			double[][] distances, int targetDimension) throws Exception {
		if (dim > maxDim) {
			throw new MyException("Dimension of the convexity test is too big!!");
		} else {
			DimensionalityReduction.runConvexityTest = true;
			((PointCloudViewableOptimizableFunction) function).setDistances(distances);

			int numValues = 100;
			double range = 2;
			computeValues2DConvexityTest(function, numValues, range, initial, dim, targetDimension, filename);

		}
	}

	/**
	 * Function to compute a 2D plot of function values of the used
	 * OptimizableFunction. The point number dim is moved in a range of "range"
	 * around its initial position. The resolution is numValues. If the filename
	 * is not an empty string the result will be saved.
	 * 
	 * @param function
	 *            used Optimizable function
	 * @param numValues
	 *            resolution of the plot
	 * @param range
	 *            range in which point number dim is moved around its initial
	 *            position
	 * @param initial
	 *            initial position of the points
	 * @param dim
	 *            number of the point that is being moved
	 * @param targetDimension
	 *            target dimension of the optimization
	 * @param filename
	 *            filename, empty -> not saved, otherwise the plot will be saved
	 *            as a .txt file under the given filename
	 */
	static void computeValues2DConvexityTest(OptimizableFunction function, int numValues, double range,
			double[] initial, int dim, int targetDimension, String filename) {
		double[] result = new double[numValues];
		double[] x = new double[numValues];

		double[] twoDplotcopy = initial.clone();
		twoDplotcopy[targetDimension * dim] -= 0.5 * range;
		for (int i = 0; i < numValues; ++i) {
			x[i] = -range / 2.0 + (range / numValues) * i;
			result[i] = function.evaluate(twoDplotcopy, 1);
			twoDplotcopy[targetDimension * dim] += ((double) range) / numValues;
		}
		save2DConvexityTest(x, result, filename);
	}

	/**
	 * Function to save and show the result of the computeValues2DConvesityTest
	 * 
	 * @param x
	 *            array of x coordinates of the plot being saved and shown
	 * @param y
	 *            array of y coordinates of the plot being saved and shown
	 * @param filename
	 *            filename, empty -> not saved, otherwise the plot will be saved
	 *            as a .txt file under the given filename
	 */
	static void save2DConvexityTest(double[] x, double[] y, String filename) {

		PlotHelper.plot2D(x, y);
		if (filename.length() != 0) {
			try {
				FileHandler.save(x, y, filename);
			} catch (IOException e) {
				System.err.println("cannot save ConvexityTest in file");
				e.printStackTrace();
			}
		}
		DimensionalityReduction.runConvexityTest = false;
	}

	/**
	 * Plots a 3D plot where the coordinates of the "dim" point are moved, you
	 * can see whether the function is convex in this area.
	 * 
	 * @param function
	 *            the optimization function that is used
	 * @param initial
	 *            the actual coordinates
	 * @param dim
	 *            the number of points which will be moved
	 * @param maxDim
	 *            number of points
	 * @param filename
	 *            the filename, if the result is going to be saved
	 * @param distances
	 *            matrix of inner-point distances in the original space
	 * @param targetDimension
	 *            target dimension of the mapping
	 * @throws Exception
	 */
	public static void convexityTest3D(OptimizableFunction function, double[] initial, int dim, int maxDim, String filename,
			double[][] distances, int targetDimension) throws Exception {
		if (dim > maxDim) {
			throw new MyException("Dimension of the convexity test is too big!!");
		}
		if (targetDimension <= 1) {
			throw new MyException("for a 2D convexity Test, the target dimension has to be bigger of equat to 2!");
		}
		((PointCloudViewableOptimizableFunction) function).setDistances(distances);
		DimensionalityReduction.runConvexityTest = true;
		int numValues = 100;
		double range = 1;
		computeValuesConexityTest3D(function, numValues, range, initial, dim, targetDimension, filename);
	}

	/**
	 * Function to compute a 3D plot of function values of the used
	 * OptimizableFunction. The point number dim is moved in a range of "range"
	 * around its initial position. The resolution is numValues. If the filename
	 * is no empty string the result will be saved.
	 * 
	 * @param function
	 *            used Optimizable function
	 * @param numValues
	 *            resolution of the plot
	 * @param range
	 *            range in which point number dim is moved around its initial
	 *            position
	 * @param initial
	 *            initial position of the points
	 * @param dim
	 *            number of the point that is being moved
	 * @param targetDimension
	 *            target dimension of the optimization
	 * @param filename
	 *            filename, empty -> not saved, otherwise the plot will be saved
	 *            as a .txt file under the given filename
	 */
	static void computeValuesConexityTest3D(OptimizableFunction function, int numValues, double range, double[] initial,
			int dim, int targetDimension, String filename) {

		double[] x = new double[numValues];
		double[] y = new double[numValues];
		double[][] z = new double[numValues][numValues];
		double[] copy = initial.clone();
		copy[targetDimension * dim] -= 0.5 * range;
		copy[targetDimension * dim + 1] -= 0.5 * range;
		double copyFirstCoordinate = copy[targetDimension * dim + 1];
		for (int i = 0; i < numValues; ++i) {
			x[i] = -0.5 * range + i * (range / numValues);
			copy[targetDimension * dim + 1] = copyFirstCoordinate;
			copy[targetDimension * dim] += ((double) range / numValues);
			for (int j = 0; j < numValues; ++j) {
				y[i] = -0.5 * range + i * (range / numValues);
				z[i][j] = function.evaluate(copy, 1);
				copy[targetDimension * dim + 1] += ((double) range / numValues);
			}

		}
		save3DConvexityTest(x, y, z, range, filename);

	}

	/**
	 * Function to save and show the result of the computeValues3DConvesityTest
	 * 
	 * @param x
	 *            array of x coordinates of the plot being saved and shown
	 * @param y
	 *            array of y coordinates of the plot being saved and shown
	 * @param z
	 *            array of z coordinates of the plot being saved and shown
	 * @param filename
	 *            filename, empty -> not saved, otherwise the plot will be saved
	 *            as a .txt file under the given filename
	 */
	static void save3DConvexityTest(double[] x, double[] y, double[][] z, double range, String filename) {
		PlotHelper.plot3D(x, y, z);

		if (filename.length() != 0) {
			double[] xSave = new double[x.length * y.length];
			double[] ySave = new double[x.length * y.length];
			double[] zSave = new double[x.length * y.length];
			int pos = 0;
			for (int i = 0; i < x.length; ++i) {
				for (int j = 0; j < y.length; ++j) {
					xSave[pos] = x[i];
					ySave[pos] = y[j];
					zSave[pos] = z[i][j];
					++pos;

				}
			}
			try {
				FileHandler.save(xSave, ySave, zSave, filename);
			} catch (IOException e) {
				System.err.println("cannot save 3D ConvexityTest in file");
				e.printStackTrace();
			}
		}
		DimensionalityReduction.runConvexityTest = false;
	}

}
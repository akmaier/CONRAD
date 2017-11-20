package edu.stanford.rsl.conrad.dimreduction;


import java.util.Random;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.dimreduction.utils.Error;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;


/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class LagrangianOptimization {

	static double[] bestIni = null;
	static double minError = -1.0;
	private double[][] distances;
	private int targetDimension;
	private DimensionalityReduction dimRed;
	private PointCloudViewableOptimizableFunction function;
	private double[] bestTimeValue;
	private double minimumError;
	private double[] iterations;
	private double[] errors;

	/**
	 * constructor for a LagrangianOptimization
	 * 
	 * @param distances
	 *            distance matrix of the high-dimensional points
	 * @param targetDimension dimention of the target space/low dimensional space
	 * @param dimRed the used DimensionalityReduction
	 * @param function the according PoitCloudViewableOptimizabaleFunction
	 */
	public LagrangianOptimization(double[][] distances, int targetDimension,
			DimensionalityReduction dimRed, PointCloudViewableOptimizableFunction function) {
		this.distances = distances;
		this.targetDimension = targetDimension;
		this.dimRed = dimRed;
		this.function = function;
		function.setOptimization(dimRed);
	}
	
	public double[] optimize(double[] initial) throws Exception {
		return optimize(initial, true);
	}

	/**
	 * optimizes the function with parameters, determined in SammonTest and the
	 * GUI
	 * 
	 * @param initial
	 *            double array of the starting points
	 * @return double array of the computed coordinates
	 * @throws Exception
	 */
	public double[] optimize(double[] initial, boolean showPoints) throws Exception {

		PointCloudViewableOptimizableFunction pcFunction = null;
		try {
			pcFunction = (PointCloudViewableOptimizableFunction) function;
			pcFunction.setOptimization(dimRed);
			pcFunction.setDistances(distances);

		} catch (ClassCastException e) {
			System.err.println(e.getStackTrace());
		}



		FunctionOptimizer func = new FunctionOptimizer();
		func.setDimension(this.distances.length * this.targetDimension);

		if (dimRed.getImprovement() != null
				&& dimRed.getImprovement().getRestriction()) {
			double[] min = new double[this.distances.length
			                          * this.targetDimension];
			double[] max = new double[this.distances.length
			                          * this.targetDimension];
			double lowerBound = dimRed.getImprovement().getLowerBound();
			double upperBound = dimRed.getImprovement().getUpperBound();
			for (int i = 0; i < this.distances.length * this.targetDimension; ++i) {
				min[i] = lowerBound;
				max[i] = upperBound;
			}

			func.setMinima(min);
			func.setMaxima(max);
		}
		func.setConsoleOutput(false);

		if (showPoints) {PointCloudViewer pc = HelperClass.showPoints(initial,
				this.distances.length, "Current Mapping");
		if (pcFunction != null)
			pcFunction.setPointCloudViewers(pc);
		}
		func.setInitialX(initial);
		func.setOptimizationMode(OptimizationMode.Function);
		function.setDistances(distances);
		func.optimizeFunction(function);

		if (dimRed.getBestTimeValue()) {
			bestTimeValue = pcFunction.getBestX();
			minimumError = pcFunction.getMinError();
		}
		if (dimRed.getPlotIterError() != null) {
			iterations = pcFunction.getIterations();
			errors = pcFunction.getErrors();
		}

		return func.getOptimum();

	}

	/**
	 * returns the best result of all iteration steps
	 * 
	 * @return the best result over the time. Fetches the result from the
	 *         PointCloudViewable function
	 */
	double[] getBestTimeValue() {
		return bestTimeValue;
	}

	/**
	 * returns the minimum error of all iteration steps
	 * 
	 * @return the minimum error of all iteration steps
	 */
	double getMinimumError() {
		return minimumError;
	}

	/**
	 * returns an array of errors from every iteration step of the gradient descent
	 * 
	 * @return an array of errors from every iteration step of the gradient
	 *         descent
	 */
	public double[] getErrors() {
		return errors;
	}

	/**
	 * returns an array of number of iterations, where the error was computed
	 * 
	 * @return an array of the number of iterations, where the error was
	 *         computed
	 */
	public double[] getIterations() {
		return iterations;
	}

	/**
	 * computes a random initialization with the right number of coordinates. If
	 * you initialize more than once, the best initialization, with the smallest
	 * sammon error will be stored:
	 * 
	 * @return a double array of random points in a range of [-0.5,
	 *         0.5]^{dimension * numPoints}
	 */
	public double[] randomInitialization(long seed) {

		if (bestIni == null) {
			bestIni = new double[distances.length * this.targetDimension];
		}
		Random rand = new Random(seed);
		double[] initial = new double[distances.length * this.targetDimension];
		for (int i = 0; i < initial.length; i++) {
			initial[i] = rand.nextDouble() - 0.5;
		}
		Error error = new Error();
		double actualError = error.computeError(
				HelperClass.wrapArrayToPoints(initial, distances.length),
				"random initialization", distances);
		if (minError == -1.0 || actualError < minError) {
			minError = actualError;
			bestIni = initial;
		}
		return initial;
	}



	/**
	 * returns the best random initialization
	 * 
	 * @return the best random initialization
	 */
	public double[] bestIni() {
		System.out.println("Best Initialization found!");
		return bestIni;
	}

	public double[] randomInitialization() {
		return randomInitialization(System.currentTimeMillis());
	}

}
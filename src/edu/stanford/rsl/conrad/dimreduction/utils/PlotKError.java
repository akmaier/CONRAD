package edu.stanford.rsl.conrad.dimreduction.utils;


import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.WeightedInnerProductObjectiveFunction;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class PlotKError {

	public static double minError = -1.0;
	public static double[] bestResult;

	private double interval;
	private double steplength;

	private boolean returnBestResult;

	private String filename = "";
	private int versionWIPOF = 2; 

	/**
	 * Constructor for PlotKError
	 * @param intervalEnd end of the inverval of k (start is always 0)
	 * @param steplength steplength of k
	 * @param returnBestResult boolean, whether the best result of all tried k should be displayed
	 * @param version version of the WeightedInnerProductObjectiveFunction
	 */
	public PlotKError(double intervalEnd, double steplength, boolean returnBestResult, int version) {
		this.interval = intervalEnd;
		this.steplength = steplength;
		this.returnBestResult = returnBestResult;
		this.versionWIPOF = version;
	}
	
	/**
	 * Constructor for PlotKError
	 * @param intervalEnd end of the inverval of k (start is always 0)
	 * @param steplength steplength of k
	 * @param returnBestResult boolean, whether the best result of all tried k should be displayed
	 * @param version version of the WeightedInnerProductObjectiveFunction
	 * @param filename if the filename is set the computed results of the plot will be saved in a txt file, (give filename without .txt!)
	 */
	public PlotKError(double intervalEnd, double steplength, boolean returnBestResult, int version, String filename) {
		this.interval = intervalEnd;
		this.steplength = steplength;
		this.returnBestResult = returnBestResult;
		this.filename = filename; 
		this.versionWIPOF = version;
	}


	/**
	 * Computes the 2D plot "k-error"
	 * 
	 * @param distanceMatrix
	 *            original distances in the high-dimensional space
	 * @throws Exception
	 */
	public void computePlot_k_error(double[][] distanceMatrix,
			DimensionalityReduction optimization) throws Exception {

		double[] errors = new double[(int) Math.ceil((double) interval
				/ steplength)];
		double[] k = new double[(int) Math.ceil((double) interval / steplength)];

		int counter = 0;
		for (double i = 0; i < interval; i += steplength) {
			PointCloudViewableOptimizableFunction functionScatter = new WeightedInnerProductObjectiveFunction(versionWIPOF, i);
			((WeightedInnerProductObjectiveFunction) functionScatter)
					.setDistances(distanceMatrix);
			functionScatter.setDistances(distanceMatrix);
			
			functionScatter.setOptimization(optimization);
			FunctionOptimizer func = new FunctionOptimizer();
			int newDim = distanceMatrix.length
					* optimization.getTargetDimension();
			func.setDimension(newDim);
			double[] initial = new double[newDim];
			for (int j = 0; j < initial.length; j++) {
				initial[j] = Math.random() - 0.5;

			}

			func.setConsoleOutput(false);

			func.setInitialX(initial);
			func.setOptimizationMode(OptimizationMode.Function);
			func.optimizeFunction(functionScatter);
			double[] result = func.getOptimum();
			Error er = new Error();
			double error = er.computeError(HelperClass.wrapArrayToPoints(
					result, distanceMatrix.length), "", distanceMatrix);
			if (minError == -1.0 || minError > error) {
				minError = error;
				bestResult = result;
			}
			errors[counter] = error;
			k[counter] = i;
			++counter;

		}

		if (filename.length() != 0) {
			FileHandler.save(k, errors, filename);
		}

		PlotHelper.plot2D(k, errors);
		if (returnBestResult) {
			HelperClass.showPoints(this.getBestResult(), distanceMatrix.length,
					"Result with optimal k");
			System.out.println("Minimal error with best k: "
					+ this.getMinError());
		}

	}

	/**
	 * returns the minimal error of all used k-factors
	 * 
	 * @return returns the minimal error of all used k-factors after a "k-error"
	 *         plot
	 */
	public double getMinError() {
		return minError;
	}

	/**
	 * returns the coordinates of the result with the smallest Sammon error
	 * 
	 * @return the coordinates of the result with the smallest Sammon error
	 *         after a "k-error" plot
	 */
	public double[] getBestResult() {
		return bestResult;
	}

}
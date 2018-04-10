package edu.stanford.rsl.conrad.dimreduction.utils;


import java.util.ArrayList;



import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.jpop.OptimizableFunction;

/*
 * Copyright (C) 2013-14  Susanne Westphal, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public abstract class PointCloudViewableOptimizableFunction implements
		OptimizableFunction {

	PointCloudViewer pointCloudViewers;
	protected DimensionalityReduction dimRed;
	int interval = 100;
	int count = 0;
	private static ArrayList<Double> errors = new ArrayList<Double>();
	private static ArrayList<Double> iterations = new ArrayList<Double>();

	private static double errorMin = -1.0;
	private static double[] bestX = null;
	private static ArrayList<Double> distMeanValue = new ArrayList<Double>();

	
	/**
	 * 
	 * @return the update interval of the pointCloudViewer
	 */
	public int getInterval() {
		return interval;
	}
	
	/**
	 * sets the update interval of the pointCloudViewer
	 * 
	 * @param interval
	 */
	public void setInterval(int interval) {
		this.interval = interval;
	}

	/**
	 * returns the PointCloudViewer
	 * 
	 * @return the PointCloudViewer
	 */
	public PointCloudViewer getPointCloudViewers() {
		return pointCloudViewers;
	}

	/**
	 * sets the point cloud viewer
	 * 
	 * @param pointCloudViewers
	 */
	public void setPointCloudViewers(PointCloudViewer pointCloudViewers) {
		this.pointCloudViewers = pointCloudViewers;
	}

	
	/**
	 * sets the distance matrix in the Objective functions, nothing to do here
	 * @param distances
	 */
	public void setDistances(double[][] distances){
		
	}

	private class SetterRun implements Runnable {

		double[] x;
		
		public SetterRun(double[] x) {
			this.x = x;
			
		}
		
		public void run() {
			if (pointCloudViewers != null)
				pointCloudViewers.setPoints(HelperClass.wrapArrayToPoints(x,
						x.length / dimRed.getTargetDimension()));			
		}

	}
	
	/**
	 * function to set the DimensionalityReduction
	 * @param dimRed DimensionalityReduction to be set
	 */
	public void setOptimization(DimensionalityReduction dimRed){
		this.dimRed = dimRed; 
	}
	
	

	/**
	 * computes every interval iterations: the mean value of the coordinates,
	 * the actual error, the smallest error until now and the corresponding
	 * coordinates and saves all in ArrayLists
	 * 
	 * @param x
	 *            actual coordinates of the optimization
	 * @param dimRed DimensionalityReduction of the current optimization
	 */
	protected void updatePoints(double[] x, DimensionalityReduction dimRed) {
		if(this.dimRed == null){
			this.dimRed = dimRed;
			System.out.println("set dimred");
		}
		count++;
		if ((count % interval) == 0) {

			Thread t = new Thread(new SetterRun(x));
			t.start();
			

			double sum1 = 0.0;
			double sum2 = 0.0;
			for (int i = 0; i < x.length / 2; ++i) {
				sum1 += x[i * 2];
				sum2 += x[i * 2 + 1];
			}

			double abs = (sum1 / (x.length / 2)) * (sum1 / (x.length / 2))
					+ (sum2 / (x.length / 2)) * (sum2 / (x.length / 2));
			distMeanValue.add(Math.sqrt(abs));

			// computes every interval iteration the Sammon Error
			if (dimRed.getPlotIterError() != null
					|| dimRed.getBestTimeValue()) {
				Error error = new Error();
				double actualError = error.computeError(
						HelperClass.wrapArrayToPoints(x, x.length
								/ dimRed.getTargetDimension()),
						"timefunction", dimRed.getDistances());
				if (errorMin == -1.0 || errorMin > actualError) {
					errorMin = actualError;
					if (bestX == null) {
						bestX = new double[x.length];
					}
					bestX = x;
				}

				errors.add(actualError);
				iterations.add((double) count);
			}
		}
	}

	/**
	 * 
	 * @return an array with the mean distance of the points to the origin every
	 *         interval iterations
	 */
	public static double[] getDistMean() {
		int length = distMeanValue.size();
		double[] distMeanValues = new double[length];
		for (int i = 0; i < length; ++i) {
			distMeanValues[i] = distMeanValue.get(i);
		}
		return distMeanValues;

	}

	/**
	 * returns the coordineats of the points with the smallest Sammon Error
	 * 
	 * @return the coordinates of the points with the smallest Sammon Error
	 */
	public double[] getBestX() {
		return bestX;
	}

	/**
	 * returns the minimum error of all iterations steps
	 * 
	 * @return the minimum error of all iteration steps
	 */
	public double getMinError() {
		return errorMin;
	}

	/**
	 * returns an array with the Sammon Errors of all interval iteration steps
	 * 
	 * @return an array with the Sammon Errors of all interval iteration steps
	 */
	public double[] getErrors() {
		int length = errors.size();
		double[] error = new double[length];
		for (int i = 0; i < length; ++i) {
			error[i] = errors.get(i);
		}
		return error;
	}

	/**
	 * returns an array with the numbers of iterations with a step length of interval
	 * 
	 * @return an array with the numbers of iterations with a step length of
	 *         interval (default is 100)
	 */
	public double[] getIterations() {
		int length = iterations.size();
		double[] iteration = new double[length];

		for (int i = 0; i < length; ++i) {
			iteration[i] = iterations.get(i);
		}

		return iteration;
	}

}

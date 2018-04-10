package edu.stanford.rsl.conrad.dimreduction.utils;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.LagrangianOptimization;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class OptimizationHelper {

	static int numberOfIterations = 1000;

	/**
	 * Computes numberOfIterations random initializations and returns the best
	 * one
	 * 
	 * @param distances
	 *            Distance matrix
	 * @param targetDimension
	 *            target dimension
	 * @param optimization
	 *            running optimization
	 * @return the best random initialization
	 */
	public static double[] optimizeWithBestInitialization(double[][] distances,
			int targetDimension, DimensionalityReduction optimization) {
		LagrangianOptimization laop = new LagrangianOptimization(distances,
				targetDimension, optimization, (PointCloudViewableOptimizableFunction) optimization.getTargetFunction());
		for (int i = 0; i < numberOfIterations; ++i) {
			laop.randomInitialization();
		}
		return laop.bestIni();

	}

}
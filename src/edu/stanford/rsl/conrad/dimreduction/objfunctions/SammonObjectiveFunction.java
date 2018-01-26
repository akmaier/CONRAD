package edu.stanford.rsl.conrad.dimreduction.objfunctions;


import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class SammonObjectiveFunction extends PointCloudViewableOptimizableFunction implements GradientOptimizableFunction {

	private double[][] distanceMap;
	private DimensionalityReduction dimRed; 
	double s;
	/**
	 * sets the DimensionalityReduction
	 * @param optimization
	 */
	public SammonObjectiveFunction() {
	}
	
	public void setOptimization(DimensionalityReduction dimRed){
		this.dimRed = dimRed;
	}
	
	/**
	 * sets the distance matrix for the mapping
	 * @param distances
	 */
	public void setDistances(double[][] distances) {
		distanceMap = distances;
		s = 0;
		for (int p = 0; p < distanceMap.length; p++) {
			for (int q = p + 1; q < distanceMap[0].length; q++) {
				s += distanceMap[p][q];
			}
		}
		s = 1.0 / s;
	
	}

	@Override
	public void setNumberOfProcessingBlocks(int number) {

	}

	@Override
	public int getNumberOfProcessingBlocks() {
		return 1;
	}

	/**
	 * Sammon Objective Function
	 * \left(\sum_p \sum_{q > p} \frac{(d_{pq} - ||x_p - x_q||_2)^2}{d_{pq}}\right) \frac{1}{\sum_p \sum_{q > p}d_{pq}}
	 */
	public double evaluate(double[] x, int block) {
		
		if (!DimensionalityReduction.runConvexityTest) {
			updatePoints(x, dimRed);
		}
		double value = 0;
		for (int p = 0; p < distanceMap.length; p++) {
			for (int q = p + 1; q < distanceMap[0].length; q++) {
				value += Math.pow(distanceMap[p][q] - HelperClass.distance(x, p, q, dimRed.getTargetDimension()), 2)
						/ distanceMap[p][q];
			}
		}
		return value * s;
	}



	/**
	 * computes the Euclidean distance in vector x of elements p and q for 2D
	 * vectors.
	 * 
	 * @param x
	 * @param p
	 * @param q
	 * @return the distance.
	 */
	double distance2D(double[] x, int p, int q) {
		return Math.sqrt(Math.pow(x[p * 2] - x[q * 2], 2)
				+ Math.pow(x[(p * 2) + 1] - x[(q * 2) + 1], 2));
	}

	@Override
	public double[] gradient(double[] x, int block) {
		double[] gradient = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			int mod = i % 2;
			for (int p = 0; p < distanceMap.length; p++) {
				for (int q = p + 1; q < distanceMap[0].length; q++) {
					double distance = distance2D(x, p, q);
					gradient[i] = 2 * (x[(p * 2) + mod] - x[(q * 2) + mod])
							* (distanceMap[p][q] - distance) / distance;
				}
			}
			gradient[i] *= s;
		}
		return gradient;
	}

}

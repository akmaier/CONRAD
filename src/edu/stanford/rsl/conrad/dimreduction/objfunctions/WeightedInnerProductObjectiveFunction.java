package edu.stanford.rsl.conrad.dimreduction.objfunctions;


import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;

/*
 * Copyright (C) 2013-14  Susanne Westphal, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class WeightedInnerProductObjectiveFunction extends PointCloudViewableOptimizableFunction
		implements GradientOptimizableFunction {

	private double[][] distanceMap;
	private double k;
	private DimensionalityReduction dimRed;
	private int versionWIPOF;

	/**
	 * Constructor, sets the DimensionalityReduction and the default settings
	 * 
	 * @param optimization
	 */
	public WeightedInnerProductObjectiveFunction(int versionWIPOF, double k) {
		this.versionWIPOF = versionWIPOF;
		this.k = k;
	}

	

	/**
	 * sets the distance matrix
	 * 
	 * @param distances
	 *            distance matrix of the high-dimensional space
	 */
	public void setDistances(double[][] distances) {
		distanceMap = distances;
	}

	/**
	 * sets the actual k-factor
	 * 
	 * @param k
	 *            actual k-factor
	 */
	public void setK(double k) {
		this.k = k;
	}

	/**
	 * 
	 * @return the actual k-factor
	 */
	public double getK() {
		return this.k;
	}

	/**
	 * sets the version of WIPOF that is used
	 * 
	 * @param i
	 *            version of WIPOF
	 */
	public void setVersionWIPOF(int i) {
		this.versionWIPOF = i;
	}

	/**
	 * return the version of the WeightedInnerProductObjectiveFunction
	 * @return the version of the WeightedInnerProductObjectiveFunction
	 */
	public int getVersionWIPOF() {
		return this.versionWIPOF;
	}

	@Override
	public void setNumberOfProcessingBlocks(int number) {

	}

	@Override
	public int getNumberOfProcessingBlocks() {
		return 1;
	}

	/**
	 * Weighted Inner Product Objective Function version 2: \sum_p \sum_{q>p}
	 * \left(\frac{2 \cdot \left\langle \vec{x}_p, \vec{x}_q \right\rangle +
	 * d_{pq}^2}{d_{pq} + k}\right)^2 version 3: \sum_p \sum_{q>p} \frac{(2
	 * \cdot \left\langle \vec{x}_p, \vec{x}_q \right\rangle +
	 * d_{pq}^2)^2}{d_{pq} + k}
	 * @param x array of all point coordinates
	 * @param block not needed here
	 */
	public double evaluate(double[] x, int block) {

		if (!DimensionalityReduction.runConvexityTest) {
			updatePoints(x, dimRed);
		}

		double value = 0;
		for (int p = 0; p < distanceMap.length; p++) {
			for (int q = p + 1; q < distanceMap[0].length; q++) {
				double distance = HelperClass.innerProduct2D(x, p, q);

				// works only with random initialization
				if (versionWIPOF == 2) {

					value += Math.pow(
							(2 * (distance) + (distanceMap[p][q] * distanceMap[p][q])) / (k + distanceMap[p][q]), 2);

				} else if (versionWIPOF == 3) {

					value += Math.pow((2 * (distance) + (distanceMap[p][q] * distanceMap[p][q])), 2)
							/ (k + distanceMap[p][q]);

				}

			}
		}
		return Math.pow(value, 1);
	}

	@Override
	public double[] gradient(double[] x, int block) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * function to set the DimensionalityReduction
	 * 
	 * @param dimRed
	 */
	public void setDimensionalityReduction(DimensionalityReduction dimRed) {
		this.dimRed = dimRed;

	}
	
	/**
	 * sets the DimensionalityReduction
	 * @param dimRed DimensionalityReduction to set
	 */
	public void setOptimization(DimensionalityReduction dimRed) {
		this.dimRed = dimRed;
	}

}

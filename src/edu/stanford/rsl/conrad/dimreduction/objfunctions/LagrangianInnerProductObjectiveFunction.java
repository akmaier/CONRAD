package edu.stanford.rsl.conrad.dimreduction.objfunctions;


import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;

/*
 * Copyright (C) 2013-14  Susanne Westphal, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class LagrangianInnerProductObjectiveFunction extends PointCloudViewableOptimizableFunction implements GradientOptimizableFunction {

	
	private double[][] distanceMap;
	private DimensionalityReduction dimRed; 

	/**
	 * Constructor of the LagrangianInnerProductObjectiveFunction
	 */
	public LagrangianInnerProductObjectiveFunction(){
		// nothing to do here
	}
	

	/**
	 * sets the distance matrix
	 * @param distances
	 */
	public void setDistances(double[][] distances) {
		distanceMap = distances;
	}
	
	@Override
	public void setNumberOfProcessingBlocks(int number) {
	
	}

	@Override
	public int getNumberOfProcessingBlocks() {
		return 1;
	}

	/**
	 * LagrangianInnerProductObjectiveFunction : \sum_p \sum_q \left\langle \vec{x}_p, \vec{x}_q \right\rangle \cdot (2\cdot \left\langle \vec{x}_p, \vec{x}_q \right\rangle + d_{pq}^2)
	 */
	public double evaluate(double[] x, int block) {
		if(!DimensionalityReduction.runConvexityTest){
			updatePoints(x, dimRed);
		}
		double value = 0;
		for (int p=0; p < distanceMap.length;p++){
			for (int q=0; q < distanceMap[0].length;q++){
				double distance = HelperClass.innerProduct2D(x, p, q);
				double l = distance;
				// works only with random initialization
				value+= Math.pow(l*(2*distance +(distanceMap[p][q]*distanceMap[p][q])),1) ;
			}
		}
		return Math.pow(value,1);
	}

	@Override
	public double[] gradient(double[] x, int block) {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * function to set the DimensionalityReduction
	 */
	public void setOptimization(DimensionalityReduction dimRed){
		this.dimRed = dimRed;
	}
	
}

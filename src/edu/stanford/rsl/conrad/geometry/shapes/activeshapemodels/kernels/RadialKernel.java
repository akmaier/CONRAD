/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels;

import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Spherical kernel
 * @author Mathias Unberath
 *
 */
public class RadialKernel implements KernelFunction{
	
	private double theta = 1;
	static final double threeOverTwo = 3/2;

	/**
	 * Spherical kernel
	 * @param theta
	 */
	public RadialKernel(double theta){
		this.theta = theta;
	}
		
	@Override
	public float evaluateKernel(SimpleVector x, SimpleVector y) {
		float ret = 0;
		double val = SimpleOperators.subtract(x, y).normL2() / theta;
		if( val > 1){
			return ret;
		}else{
			ret = (float)(1 - threeOverTwo * val + 0.5 * Math.pow(val,3));
		}
		return ret;
	}

	@Override
	public String getName() {
		return "RadialKernel";
	}

	

}

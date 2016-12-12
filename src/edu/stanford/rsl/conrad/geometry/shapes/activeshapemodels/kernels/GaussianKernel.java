/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels;

import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class GaussianKernel implements KernelFunction{

	/**
	 * The standard deviation of the Gaussian kernel i.e. the support of the radial kernel.
	 */
	double sigma;
	
	public GaussianKernel(double sigma){
		this.sigma = sigma;
	}
	
	public GaussianKernel(){
		this.sigma = 1.0;
	}

	@Override
	public float evaluateKernel(SimpleVector x, SimpleVector y) {
		
		SimpleVector diff = x.clone();
		diff.subtract(y);
		double val = - Math.pow(diff.normL2(),2);
		val /= (2 * Math.pow(sigma, 2));
		val = Math.exp( val );
		
		return (float)val;
	}

	@Override
	public String getName() {
		String name = "Gaussian " + Double.valueOf(sigma);
		return name;
	}
	
	
	
}

/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels;

import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class PolynomialKernel implements KernelFunction{

	int order = 2;
	double alpha = 1;
	double offs = 0;
	
	public PolynomialKernel(){};
	
	public PolynomialKernel(int polynomialOrder, double slope, double offs){
		this.order = polynomialOrder;
		this.alpha = slope;
		this.offs = offs;
	}
	
	@Override
	public float evaluateKernel(SimpleVector x, SimpleVector y) {
		return (float)Math.pow(alpha * SimpleOperators.multiplyInnerProd(x, y) + offs, order);
	}

	@Override
	public String getName() {
		String name = "Polynomial " + Double.valueOf(order) +" " + Double.valueOf(alpha) + " " + Double.valueOf(offs);
		return name;
	}

	
}

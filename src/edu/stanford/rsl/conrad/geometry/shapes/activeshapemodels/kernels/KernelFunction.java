/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels;

import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This interface is used for the creation of kernel functions for the use in Kernel Principal Component Analysis.
 * @author Mathias Unberath
 *
 */
public interface KernelFunction {
	/**
	 * Evaluate the kernel function for a pair of data sets x and y.
	 * Evaluating the kernel function is equivalent to calculating the scalar-product (inner- or dot-product) in feature space.
	 * $\phi(x)^T \phi(y) = k(x,y) $
	 * @param x 
	 * @param y
	 * @return Value of dot-product in feature space
	 */
	float evaluateKernel(SimpleVector x, SimpleVector y);

	String getName();
}

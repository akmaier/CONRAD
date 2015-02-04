package edu.stanford.rsl.conrad.optimization;

import edu.stanford.rsl.conrad.data.numeric.NumericGrid;

/**
 * This interface needs to be implemented for functions that are being optimized with the LMA class.
 * @author Mathias Unberath
 *
 */
public interface OptimizableFunction {

	/**
	 * Getter for the number of parameters of the multi-dimensional optimizable function.
	 * @return The number of parameters to optimize.
	 */ 
	int getNumberOfParameters();
	
	/**
	 * Getter for the initial set of parameters for the optimization.
	 * @return The initial set of parameters.
	 */
	double[] getInitialParameters();
		
	/**
	 * Calculates the derivative of the function with respect to parameter at index dP at the point at index idx.
	 * @param data Data array containing the function values to be evaluated
	 * @param idx The index of the point to be evaluated
	 * @param param The parameter vector
	 * @param dP The component w.r.t. which the derivative shall be computed
	 * @return The derivative w.r.t. dP at x
	 */
	double getDerivativeAtPoint(NumericGrid data, int[] idx, double[] param, int dP);
	
	/**
	 * Evaluates the function at location idx using the parameters passed.
	 * @param data Data array containing the function values to be evaluated
	 * @param idx The index of the point to be evaluated
	 * @param param Current parameters
	 * @return The function's value at x
	 */
	double evaluateAtPoint(NumericGrid data, int[] idx, double[] param);
	
	/**
	 * Calculates the weights used by the Levenberg-Marquardt algorithm.
	 * @return
	 */
	NumericGrid getWeights();
	
	
	/**
	 * Calculates an element of the Jacobian of the function with respect to certain parameters.
	 * Unfortunately, this has to be implemented in the functions themselves to allow for more use cases.
	 *
	 * General procedure: pseudo code
	 * 
	 * for all points
	 *   sum += weight * function.derive(atPoint, param, wrtComponent1) * function.derive(atPoint, param, wrtComponent2)
	 * end
	 * if diagonal: sum *= (1 + lambda)
	 *  
	 * @param data The input data.
	 * @param parameters The parameters used to evaluate.
	 * @param dP1 The index of the first parameter with respect to which the derivative shall be computed.
	 * @param dP2 The index of the second parameter with respect to which the derivative shall be computed.
	 * @param weights The weights for each data point
	 * @return The partial derivative.
	 */
	@Deprecated
	public double getJacobianElement(NumericGrid data, double[] parameters, int dP1, int dP2, NumericGrid weights);
	
	/**
	 * Calculates the right-hand-side of the Levenberg-Marquardt equation for one parameter.
	 * Unfortunately, this has to be implemented in the functions themselves to allow for more use cases.
	 * 
	 * General procedure: pseudo code
	 * 
	 * for all points
	 *   sum += weight * function.derive(atPoint, param, wrtComponent) * ( values(atPoint) - function.evaluate(atPoint, param) )
	 * end
	 * 
	 * @param values The measured values to be approximated.
	 * @param data The input data.
	 * @param parameters The parameters used to evaluate.
	 * @param dP The index of the parameter with respect to which the derivative shall be computed.
	 * @param weights The weights for each data point
	 * @return The partial derivative.
	 */
	@Deprecated
	public double getBetaElements(NumericGrid values, NumericGrid data, double[] parameters, int dP, NumericGrid weights);
	
	
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
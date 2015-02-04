package edu.stanford.rsl.conrad.optimization;

import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * This class implements the Levenberg-Marquardt algorithm for the optimization of functions.
 * Derivatives with respect to each parameter have to be defined in the OptimizableFunction.
 * @author Mathias Unberath
 */
public class LMA {
	private boolean PRINT = true;
	
	/**
	 * Flag to decide whether or not numerical derivatives shall be computed.
	 */
	private boolean NUMERICAL_DERIVATIVE = false;
	/**
	 * Step-size for the computation of numerical derivatives.
	 */
	private double step  = 1e-7;
	
	/**
	 * The function used to transform the data points when approximating the values.
	 */
	private OptimizableFunction func;
	
	/**
	 * The measured function values that will be approximated.
	 * values = f(data, parameters)
	 */
	private NumericGrid values;
	
	/**
	 * The data that will be transformed to approximate the values.
	 */
	private NumericGrid data;
	/** Dimension of the Grid. */
	private int[] dim;
	
	/**
	 * Weights for each data point. 
	 * Objective function: chi2 = sum[ ( values - f(data,parameters) )^2 * weights ]
	 */
	private NumericGrid weights;
	
	/**
	 * Number of parameters of the transformation function.
	 */
	private int nParam;
	
	/**
	 * The parameters of the transforming function.
	 */
	private double[] parameters;
	/**
	 * Parameters after update.
	 */
	private double[] parametersIncremented;
	
	/** Value of objective function chi2 = sum[ ( values - f(data,parameters) )^2 * weights ] */
	private double chi2;
	/** Value of objective function chi2 after the update step. */
	private double chi2Incremented;
	
	/** Iteration counter */
	private int nIter;
	
	/** Jacobian matrix w.r.t the parameters. */
	private SimpleMatrix alpha;
	private SimpleVector beta;
	/** Update steps for parameters. */
	private SimpleVector dParam;
	/** Damping factor. */
	private double lambda = 0.0001;
	/** Update factor for damping parameter lambda, lambda is multiplied or divided depending on 
	 * whether or not the update was successful. */
	private double lambdaFactor = 10;
	
	/** Minimal error update, end condition */
	private double deltaChi2Min = 1e-10;
	/** Maximum number of iterations. */
	private int maxIter = 100;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	public LMA(NumericGrid values, NumericGrid data, OptimizableFunction func, boolean NUMERICAL_DERIVATIVE){
		this.NUMERICAL_DERIVATIVE = NUMERICAL_DERIVATIVE;
		
		this.values = values;
		this.data = data;
		this.func = func;
		
		this.dim = values.getSize();
		assert(dim == data.getSize()) : new IllegalArgumentException("Sizes of data and value grids must match!");
		
		this.nParam = func.getNumberOfParameters();
		this.parameters = func.getInitialParameters();
		this.parametersIncremented = new double[nParam];
		
		this.weights = func.getWeights();
		
		this.alpha = new SimpleMatrix(nParam, nParam);
		this.beta = new SimpleVector(nParam);
		this.dParam = new SimpleVector(nParam);
	}
	
	/**
	 * Sets the step size for the calculation of the numerical derivative.
	 * @param step The step size to be used.
	 */
	public void setDerivativeStepSize(double step){
		this.step = step;
	}
	
	/** 
	 * Returns the parameters after optimization.
	 * @return The final parameters
	 */
	public double[] getFinalParameters(){
		return this.parameters;
	}
	
	/**
	 * Set the parameters and perform optimization. If the parameters are set to be smaller or equal to 0, the default values will be used.
	 * @param lambda The step width.
	 * @param deltaChi2Min The stop condition for the update.
	 * @param maxIter The stop condition for number of iterations.
	 */
	public void run(double lambda, double deltaChi2Min, int maxIter){
		System.out.println();
		System.out.println("Starting Levenberg-Marquardt optimization.\n");
		if(lambda <= 0){
			System.out.println("Using default value for lambda: " + this.lambda);
		}else{
			this.lambda = lambda;
			System.out.println("Lambda is: " + this.lambda);
		}
		
		if(deltaChi2Min <= 0){
			System.out.println("Using default value for minimum update step: " + this.deltaChi2Min);
		}else{
			this.deltaChi2Min = deltaChi2Min;
			System.out.println("Minimum update step is: " + this.deltaChi2Min);
		}
		
		if(maxIter <= 0){
			System.out.println("Using default value for maximum number of iterations: " + this.maxIter);
		}else{
			this.maxIter = maxIter;
			System.out.println("Maximum number of iterations is: " + this.maxIter);
		}
		
		try {
			runInternal();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Performs the optimization. Is called by the public run method that allows for the setting of different optimization parameters.
	 * @throws Exception 
	 */
	private void runInternal() throws Exception{
		
		nIter = 0;
		// check if input parameters are valid
		chi2 = getChi2();
		if(Double.isNaN(chi2)){
			throw new IllegalArgumentException("Initial function parameters seem to be invalid.");
		}
		
		// perform optimization
		while( !stop() ){
			chi2 = getChi2();
			
			updateAlpha();
			updateBeta();
			
			if(PRINT){
				String param ="";
				for(int ipar = 0; ipar < nParam; ipar ++){
					param += Double.valueOf(parameters[ipar]) + " ";
				}
				System.out.println(nIter + " : " + chi2 + " : " + param);
			}
			
			try{
				getIncrements();
			
				chi2Incremented = getChi2Incremented();
				// if update was worse than initial guess, increase damping, e.g. smaller step
				if( (chi2Incremented >= chi2) || (Double.isNaN(chi2Incremented)) ){
					lambda *= lambdaFactor;
				}else{
					// if update was successful decrease damping, e.g. bigger step
					lambda /= lambdaFactor;
					updateParameters();					
				}
			}catch(Exception e){
				// only throw exception if it was the last step
				// if intermediate step, try increased damping
				if(nIter == maxIter
						){
					e.printStackTrace();
					throw e;
				}else{
					lambda *= lambdaFactor;
				}
					
			}
			nIter++;
		}
		
		printResults();		
	}
	
	/**
	 * Calculates all elements of the beta vector and sets them at the corresponding position.
	 */
	private void updateBeta(){
		for(int i = 0; i < nParam; i++){
			this.beta.setElementValue(i, getBetaElement(i));
		}
	}
	
	/** 
	 * Calculates an entry of the beta vector. This vector is the right-hand-side of the Levenberg-Marquardt function.
	 * Used to be calculated in the OptimizableFunction. This approach is still possible if changing the code to call the function's method,
	 * however we do not recommend this.
	 * General procedure: pseudo code
	 * 
	 * for all points
	 *   sum += weight * function.derive(atPoint, param, wrtComponent) * ( values(atPoint) - function.evaluate(atPoint, param) )
	 * end
	 * 
	 * @param row Index of derivative parameter
	 * @return The beta element for this parameter
	 */
	private double getBetaElement(int row){
		int nVal = this.values.getNumberOfElements();
		double[] dF = new double[nVal];
		for(int i = 0; i < nVal; i++){
			int[] idx = linearToArrayIndex(i);
			if(NUMERICAL_DERIVATIVE){
				dF[i] = getNumericalDerivativeAtPoint(data, idx, parameters, row);
			}else{
				dF[i] = func.getDerivativeAtPoint(data, idx, parameters, row);
			}
		}
		
		double val = 0;
		for(int i = 0; i < nVal; i++){
			int[] idx = linearToArrayIndex(i);
			double diff = values.getValue(idx) - func.evaluateAtPoint(data, idx, parameters);
			val += weights.getValue(idx) * dF[i] * diff;
		}
		return val;
	}
	
	
	/**
	 * Calculates all elements of the alpha matrix and sets them at the corresponding positions.
	 */
	private void updateAlpha(){
		for(int i = 0; i < nParam; i++){
			for(int j = 0; j < nParam; j++){
				//System.out.println("Calculating alpha: " + i + " , " + j);
				alpha.setElementValue(i, j, getAlphaValue(i, j));
			}
		}
	}
	
	/**
	 * Calculates the entry at position (row, col) in the Jacobian Matrix.
	 * Used to be calculated in the OptimizableFunction. This approach is still possible if changing the code here to call the function's method,
	 * however we do not encourage this.
	 * General procedure: pseudo code
	 * 
	 * for all points
	 *   sum += weight * function.derive(atPoint, param, wrtComponent1) * function.derive(atPoint, param, wrtComponent2)
	 * end
	 * if diagonal: sum *= (1 + lambda)
	 * 
	 * @param row The index of first derivative parameter
	 * @param col The index of second derivative parameter
	 * @return The Jacobian entry
	 */
	private double getAlphaValue(int row, int col){
		double val = getJacobianElement(data, parameters, row, col, weights);
		
		return val;		
	}
	
	/**
	 * Calculates the Jacobian element at index (row,col) in the manner specified in the calling method.
	 * @param data
	 * @param paramters
	 * @param row
	 * @param col
	 * @param weights
	 * @return The Jacobian element
	 */
	private double getJacobianElement(NumericGrid data, double[] paramters, int row, int col, NumericGrid weights){
		int nVal = this.values.getNumberOfElements();
		double[] dRow = new double[nVal];
		double[] dCol = new double[nVal];
		
		if(NUMERICAL_DERIVATIVE){
			for(int i = 0; i < nVal; i++){
				int[] idx = linearToArrayIndex(i);
				dRow[i] = getNumericalDerivativeAtPoint(data, idx, parameters, row);
			}
			for(int i = 0; i < nVal; i++){
				int[] idx = linearToArrayIndex(i);
				dCol[i] = getNumericalDerivativeAtPoint(data, idx, parameters, col);
			}
		}else{
			for(int i = 0; i < nVal; i++){
				int[] idx = linearToArrayIndex(i);
				dRow[i] = func.getDerivativeAtPoint(data, idx, parameters, row);
			}
			for(int i = 0; i < nVal; i++){
				int[] idx = linearToArrayIndex(i);
				dCol[i] = func.getDerivativeAtPoint(data, idx, parameters, col);
			}			
		}
		
		double val = 0;
		for(int i = 0; i < nVal; i++){
			int[] idx = linearToArrayIndex(i);			
			val += weights.getValue(idx) * dRow[i] *  dCol[i];
		}
		if(row == col){
			val *= (1 + lambda);
		}		
		return val;
	}
	
	/**
	 * Calculates the numerical derivative using finite differencing. We use a a symmetric difference of this.step in forward and backward direction.
	 * @param data Data array containing the function values to be evaluated
	 * @param idx The index of the point to be evaluated
	 * @param param The current parameter estimate
	 * @param dP The parameter w.r.t. which the derivative shall be computed
	 * @return The numerical derivative at this location
	 */
	private double getNumericalDerivativeAtPoint(NumericGrid data, int[] idx, double[] param, int dP){
		double[] lower = param.clone();
		lower[dP] -= this.step;
		double[] upper = param.clone();
		upper[dP] += this.step;
		
		return ( (func.evaluateAtPoint(data, idx, upper) - func.evaluateAtPoint(data, idx, lower)) / (2 * step));		
	}
	
	/**
	 * Method to compute a multi-dimensional array index from a linear index. Implemented from:
	 * http://math.stackexchange.com/questions/19765/calculating-coordinates-from-a-flattened-3d-array-when-you-know-the-size-index
	 * @param lin The linear index.	
	 * @return The multi-dimensional index.
	 */
	private int[] linearToArrayIndex(int lin){
		int[] idx = new int[this.dim.length];
		for(int i = this.dim.length-1; i >= 0; i--){
			int val = 1;
			for(int j = 0; j < i; j++){
				val *= this.dim[j];
			}
			idx[i] = lin / val;
			lin -= idx[i]*val;
		}
		return idx;
	}
	
	/**
	 * Calculates the incremented parameters using the alpha matrix. Make sure to check if inverse is well conditioned.
	 * @throws Exception 
	 */
	private void getIncrements() throws Exception{
		this.alpha = alpha.inverse(InversionType.INVERT_SVD);
		if(alpha.isSingular(CONRAD.DOUBLE_EPSILON)){
			throw new Exception("Matrix is singular.");
		}
		dParam = SimpleOperators.multiply(alpha, beta);
		
		for(int i = 0; i < nParam; i++){
			this.parametersIncremented[i] = this.parameters[i] + dParam.getElement(i);
		}
	}
	
	/**
	 * This method overwrites the parameters with the incremented parameters, if accepted. 
	 */
	private void updateParameters(){
		for(int i = 0; i < nParam; i++){
			this.parameters[i] = this.parametersIncremented[i];
		}
	}
	
	/**
	 * Prints the results after optimization to the console window.
	 */
	private void printResults(){
		System.out.println("Optimization ended after : " + nIter + " iterations.");
		System.out.println("_____________________________________________________");
		System.out.println("Goodness of fit: " + chi2 / data.getNumberOfElements());
	}
	
	/**
	 * Evaluates the convergence criterion against the forced end condition.
	 * @return Whether or not to continue.
	 */
	private boolean stop(){
		double delta = Math.abs(chi2 - chi2Incremented);
		return ( (delta < deltaChi2Min) || (nIter > maxIter) );
	}
	
	/**
	 * Returns the chi2 value for the current parameters.
	 * @return Current chi2
	 */
	private double getChi2(){
		return getChi2(this.parameters);
	}
	
	/**
	 * Returns the chi2 for the incremented parameters.
	 * @return Chi2 after applying parameter increments
	 */
	private double getChi2Incremented(){
		return getChi2(this.parametersIncremented);
	}
	
	/**
	 * Calculates chi2 for a set of parameters using the function.
	 * @param param The parameters to evaluate the function.
	 * @return chi2
	 */
	private double getChi2(double[] param){
		int nVal = this.values.getNumberOfElements();
		
		double val = 0;
		for(int i = 0; i < nVal; i++){
			int[] idx = linearToArrayIndex(i);
			val += weights.getValue(idx) * Math.pow(values.getValue(idx) - func.evaluateAtPoint(data, idx, param), 2);
		}
		return val;
	}
	
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
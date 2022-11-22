package edu.stanford.rsl.conrad.fitting;

import java.util.Random;

/**
 * This class implements a version of RANSAC that generates numberOfTries tries to generate the model that best fits the correspondences in x and y.
 * The algorithm chooses the model that has the best model fitness, i.e., the model that matches the best to the data. A correct match is counted if the
 * distance computed between model and real data is less than epsilon.<br>
 * If no model could be determined, epsilon is increased by a factor of ten and a warning is given.
 * @author akmaier
 *
 */
public class RANSACFittedFunction extends Function {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8899279893077174756L;
	/**
	 * This is the function that RANSAC is computed on-
	 */
	protected Function baseFunction;
	/**
	 * This paremeter defines the number of tries to compute RANSAC
	 */
	protected int numberOfTries = 100000;
	/**
	 * This parameter defines the fault tolerance
	 */
	protected double epsilon = 0.0001;
	
	public RANSACFittedFunction(Function func){
		baseFunction = func;
		numberOfParameters = baseFunction.getNumberOfParameters();
	}
	
	protected double evaluateModelFittnes(double [] x, double [] y, Function func){
		double eval = 0;
		for (int i=0;i<x.length;i++){
			if (Math.abs(func.evaluate(x[i])-y[i])< epsilon) eval++;
		}
		return eval;
	}
	
	@Override
	public void fitToPoints(double[] x, double[] y) {
		int corresp = baseFunction.getMinimumNumberOfCorrespondences();
		double bestFit = 0;
		
		Function bestModel = null;
		double [] randX = new double [corresp];
		double [] randY = new double [corresp];
		Random random = new Random();
		for (int i = 0; i < numberOfTries; i++){
			try {
				Function currentTry = (Function) baseFunction.getClass().newInstance();			
				// compute next try:
				for (int j=0;j<corresp;j++){
					int index = (int) (random.nextDouble() * (x.length-1));
					randX[j] = x[index];
					randY[j] = y[index];
				}
				currentTry.fitToPoints(randX, randY);
				double fitness = evaluateModelFittnes(x, y, currentTry);
				if (fitness > bestFit) {
					bestModel = currentTry;
					bestFit = fitness;
				}
			} catch (InstantiationException e) {
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				e.printStackTrace();
			}
		}
		if ((bestModel != null)) {
			baseFunction = bestModel;
			fittingDone = true;
		} else {
			System.err.println("RANSACFittedFunction: Warning increasing epsilon to " + epsilon +"\nChoose a bigger epsilon or more tries in the future!");
			epsilon *= 1.1;
			fitToPoints(x, y);			
		}
	}

	@Override
	public double evaluate(double x) {
		return baseFunction.evaluate(x);
	}

	@Override
	public String toString() {
		return "RANSAC " + baseFunction.toString();
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		// TODO Auto-generated method stub
		return baseFunction.getMinimumNumberOfCorrespondences();
	}

	/**
	 * @param baseFunction the baseFunction to set
	 */
	public void setBaseFunction(Function baseFunction) {
		this.baseFunction = baseFunction;
	}

	/**
	 * @return the baseFunction
	 */
	public Function getBaseFunction() {
		return baseFunction;
	}

	/**
	 * @param numberOfTries the numberOfTries to set
	 */
	public void setNumberOfTries(int numberOfTries) {
		this.numberOfTries = numberOfTries;
	}

	/**
	 * @return the numberOfTries
	 */
	public int getNumberOfTries() {
		return numberOfTries;
	}

	/**
	 * @return the epsilon
	 */
	public double getEpsilon() {
		return epsilon;
	}

	/**
	 * @param epsilon the epsilon to set
	 */
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	@Override
	public double[] getParametersAsDoubleArray() {
		return baseFunction.getParametersAsDoubleArray();
	}

	@Override
	public void setParametersFromDoubleArray(double[] param) {
		baseFunction.setParametersFromDoubleArray(param);
	}

}

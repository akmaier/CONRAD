package edu.stanford.rsl.conrad.fitting;

import java.io.Serializable;


/**
 * Class to describe an abstract function that can be fiited to a set of 2D Points.
 * 
 * @author akmaier
 *
 */
public abstract class Function implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 5067981485975476424L;
	protected boolean fittingDone = false;
	protected int numberOfParameters = 0;
	public abstract double [] getParametersAsDoubleArray();
	
	
	/**
	 * Fits the function to the given input data
	 * @param x the input data
	 * @param y the output data
	 */
	public abstract void fitToPoints(double [] x, double [] y);
	
	public void fitToPoints(float []x, float []y){
		double [] dx = new double[x.length];
		double [] dy = new double[y.length];
		for (int i= 0; i < x.length; i++){
			dx[i]= x[i];
			dy[i]= y[i];
		}
		fitToPoints(dx, dy);
	}
	
	/**
	 * Evaluates the function at position x
	 * @param x the position
	 * @return the output value
	 */
	public abstract double evaluate(double x);
	
	public abstract String toString();
	
	public abstract int getMinimumNumberOfCorrespondences();
	
	public static Function[] getAvailableFunctions(){
		Function [] revan = {new LinearFunction(), new LogarithmicFunction(), new IdentityFunction()};
		return revan;
	}


	/**
	 * @return the numberOfParameters
	 */
	public int getNumberOfParameters() {
		return numberOfParameters;
	}
	
}

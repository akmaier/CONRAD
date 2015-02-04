package edu.stanford.rsl.conrad.fitting;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class PolynomialFunction extends Function {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1557807323919814217L;
	private int degree =2;
	private double [] params;
	
	public PolynomialFunction (){
		params = new double [degree +1];
	}
	
	/**
	 * @return the degree
	 */
	public int getDegree() {
		return degree;
	}

	/**
	 * Sets the degree of the polynomial function. Note that this will reset the parameter array.
	 * @param degree the degree to set
	 */
	public void setDegree(int degree) {
		this.degree = degree;
		params = new double [degree +1];
	}

	@Override
	public double[] getParametersAsDoubleArray() {
		return params;
	}

	@Override
	public void fitToPoints(double[] x, double[] y) {
		// setup observation vector
		SimpleVector observations = new SimpleVector(y);
		// setup measurement matrix:
		SimpleMatrix measurements = new SimpleMatrix(x.length, degree+1);
		for (int i = 0; i < x.length; i++){
			for (int j = 0; j <= degree; j++){
				measurements.setElementValue(i, j, Math.pow(x[i], j));
			}
		}
		SimpleMatrix inverseMeasurements = measurements.inverse(SimpleMatrix.InversionType.INVERT_SVD);
		SimpleVector parameters = SimpleOperators.multiply(inverseMeasurements, observations);
		parameters.copyTo(params);
	}

	@Override
	public double evaluate(double x) {
		double revan = 0;
		double xx = 1;
		for (int i=0; i <= degree; i++){
			revan += params[i] * xx;
			xx *= x;
		}
		return revan;
	}

	@Override
	public String toString() {
		String revan = "" + params[0] + " ";
		for (int i=1; i <= degree; i++){
			revan += params[i] + " x^" +i;
		}
		return revan;
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		return degree + 1;
	}

}

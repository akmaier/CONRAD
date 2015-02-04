package edu.stanford.rsl.conrad.fitting;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class Quadric extends Surface {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6104784792939859900L;
	private int degree =3;
	private double [] params;
	
	public Quadric (){
		params = new double [degree*degree];
	}
	
	/**
	 * @return the degree
	 */
	public int getDegree() {
		return degree;
	}


	@Override
	public double[] getParametersAsDoubleArray() {
		return params;
	}

	@Override
	public void fitToPoints(double[] x, double[] y, double [] z) {
		// setup observation vector
		SimpleVector observations = new SimpleVector(x.length);
		// setup measurement matrix:
		SimpleMatrix measurements = new SimpleMatrix(x.length, degree*degree);
		for (int i = 0; i < x.length; i++){
				measurements.setElementValue(i, 0, Math.pow(x[i], 2));
				measurements.setElementValue(i, 1, Math.pow(y[i], 2));
				measurements.setElementValue(i, 2, Math.pow(z[i], 2));
				measurements.setElementValue(i, 3, x[i]*y[i]);
				measurements.setElementValue(i, 4, x[i]*z[i]);
				measurements.setElementValue(i, 5, y[i]*z[i]);
				measurements.setElementValue(i, 6, x[i]);
				measurements.setElementValue(i, 7, y[i]);
				measurements.setElementValue(i, 8, z[i]);
				
				observations.setElementValue(i, 1);
				
			
		}
		SimpleMatrix inverseMeasurements = measurements.inverse(SimpleMatrix.InversionType.INVERT_SVD);
		SimpleVector parameters = SimpleOperators.multiply(inverseMeasurements, observations);
		parameters.copyTo(params);
	}
	
	public void fitEllipticParaboloidToPoints(float[] x, float[] y, float [] z) {
		// setup observation vector
		SimpleVector observations = new SimpleVector(x.length);
		// setup measurement matrix:
		SimpleMatrix measurements = new SimpleMatrix(x.length, degree*degree);
		for (int i = 0; i < x.length; i++){
			measurements.setElementValue(i, 0, Math.pow(x[i], 2));
			measurements.setElementValue(i, 1, Math.pow(y[i], 2));
			measurements.setElementValue(i, 2, Math.pow(z[i], 2));
			measurements.setElementValue(i, 3, x[i]*y[i]);
			measurements.setElementValue(i, 4, x[i]*z[i]);
			measurements.setElementValue(i, 5, y[i]*z[i]);
			measurements.setElementValue(i, 6, x[i]);
			measurements.setElementValue(i, 7, y[i]);
			measurements.setElementValue(i, 8, z[i]);
			
			observations.setElementValue(i, 1);
				
			
		}
		SimpleMatrix inverseMeasurements = measurements.inverse(SimpleMatrix.InversionType.INVERT_SVD);
		SimpleVector parameters = SimpleOperators.multiply(inverseMeasurements, observations);
		parameters.copyTo(params);
	}

	@Override
	public double evaluate(double x, double y, double z) {
		double revan = 0;
		
		revan = x*x*params[0] + y*y*params[1] + z*z*params[2] + x*y*params[3] + x*z*params[4] + y*z*params[5] +x*params[6] * y*params[7] + z*params[8] -1;
		return revan;
	}

	@Override
	public String toString() {
		String revan = " ";
		
		return revan;
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		return degree*degree;
	}

}
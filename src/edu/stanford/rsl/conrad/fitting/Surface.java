package edu.stanford.rsl.conrad.fitting;

import java.io.Serializable;


/**
 * Class to describe an abstract surface that can be fitted to a set of 3D Points.
 * 
 * @author Marco Boegel
 *
 */
public abstract class Surface implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -91840779787161332L;
	protected boolean fittingDone = false;
	protected int numberOfParameters = 0;
	public abstract double [] getParametersAsDoubleArray();
	
	
	/**
	 * Fits the function to the given input data
	 * @param x the input data
	 * @param y the output data
	 */
	public abstract void fitToPoints(double [] x, double [] y, double [] z);
	
	public void fitToPoints(float []x, float []y, float [] z){
		double [] dx = new double[x.length];
		double [] dy = new double[y.length];
		double [] dz = new double[z.length];
		for (int i= 0; i < x.length; i++){
			dx[i]= x[i];
			dy[i]= y[i];
			dz[i]= z[i];
		}
		fitToPoints(dx, dy, dz);
	}
	
	/**
	 * Evaluates the function at position x,y
	 * @param x the position
	 * @param y the position
	 * @return the output value
	 */
	public abstract double evaluate(double x, double y, double z);
	
	public abstract String toString();
	
	public abstract int getMinimumNumberOfCorrespondences();
	
	public static Surface[] getAvailableSurfaces(){
		Surface [] revan = {new Quadric()};
		return revan;
	}


	/**
	 * @return the numberOfParameters
	 */
	public int getNumberOfParameters() {
		return numberOfParameters;
	}
	
}

package edu.stanford.rsl.conrad.fitting;

import edu.stanford.rsl.conrad.utils.CONRAD;

public class LinearFunction extends Function {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7484751042707779396L;
	double m = 0;
	double t = 0;

	public LinearFunction (){
		numberOfParameters = 2;
	}
	
	/**
	 * Estimates a line between (x1, y1) and (x2,y2) such that <br>
	 * y1 = m * x1 + t <br>
	 * and<br>
	 * y2 = m * x2 + t<BR>
	 * The parameters are found as:<br>
	 * m = (y1-y2)/(x1-x2)<BR>
	 * and <br>
	 * t = y1 - (m * x1)
	 * @param x1 the x coordinate of point 1
	 * @param x2 the z coordinate of point 2
	 * @param y1 the y coordinate of point 1
	 * @param y2 the y coordinate of point 2
	 */
	public LinearFunction (double x1, double x2, double y1, double y2){
		numberOfParameters = 2;
		m = (y1-y2)/(x1-x2);
		t = y1 - (m * x1);
	}
	
	@Override
	public double evaluate(double x) {
		return (m * x) + t;
	}
	
	public String toString(){
		if (fittingDone){
			return "y = (" + m + "* x) + " + t;
		} else {
			return "y = (m * x) + t";
		}
	}

	@Override
	public void fitToPoints(double[] x, double[] y) {
		double meanFirst = 0;
		double meanSecond = 0;
		for (int i = 0; i < x.length; i++){
				meanFirst += x[i];
				meanSecond += y[i];
		}
		meanFirst /= x.length;
		meanSecond /= y.length;
		double nominator = 0;
		double denominator = 0;
		for (int i = 0; i < x.length; i++){
				nominator += (x[i] - meanFirst) * (y[i] - meanSecond);
				denominator += Math.pow((x[i] - meanFirst), 2);
		}
		if (denominator == 0) denominator = CONRAD.SMALL_VALUE;
		m = nominator / denominator;
		t = meanSecond - (m * meanFirst);
		fittingDone = true;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
		fittingDone = true;
	}

	public double getT() {
		return t;
	}

	public void setT(double t) {
		this.t = t;
		fittingDone = true;
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		return 2;
	}

	@Override
	public double[] getParametersAsDoubleArray() {
		return new double[]{m,t};
	}

	@Override
	public void setParametersFromDoubleArray(double[] param) {
		m = param[0];
		t = param[1];
	}

}

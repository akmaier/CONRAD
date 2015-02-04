package edu.stanford.rsl.conrad.fitting;

public class IdentityFunction extends Function {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4775769211165404450L;

	public String toString(){
		return "y = x";
	}
	
	@Override
	public double evaluate(double x) {
		return x;
	}

	@Override
	public void fitToPoints(double[] x, double[] y) {
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		return 0;
	}

	@Override
	public double[] getParametersAsDoubleArray() {
		return new double[] {};
	}

}

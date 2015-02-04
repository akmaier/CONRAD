package edu.stanford.rsl.conrad.fitting;

/**
 * Implements a Gaussian Function.
 * Fitting is not implemented yet!
 * 
 * 
 * @author akmaier
 *
 */
public class GaussianFunction extends Function {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8345414466275108572L;
	private double mu = 0;
	private double sigma = 1;
	
	public GaussianFunction(){
		numberOfParameters = 2;
	}
	
	public GaussianFunction(double mu, double sigma){
		this.mu = mu;
		this.sigma = sigma;
	}
	
	@Override
	public void fitToPoints(double[] x, double[] y) {
		throw new RuntimeException("Not implemented yet");
	}

	@Override
	public double evaluate(double x) {
		return Math.sqrt(2*Math.PI)*sigma* Math.exp(-.5*Math.pow(((mu-x)/sigma),2));
	}

	@Override
	public String toString() {
		return "N("+mu+", "+sigma+")";
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		return 2;
	}

	/**
	 * @return the mu
	 */
	public double getMu() {
		return mu;
	}

	/**
	 * @param mu the mu to set
	 */
	public void setMu(double mu) {
		this.mu = mu;
	}

	/**
	 * @return the sigma
	 */
	public double getSigma() {
		return sigma;
	}

	/**
	 * @param sigma the sigma to set
	 */
	public void setSigma(double sigma) {
		this.sigma = sigma;
	}

	@Override
	public double[] getParametersAsDoubleArray() {
		return new double[]{mu, sigma};
	}

}

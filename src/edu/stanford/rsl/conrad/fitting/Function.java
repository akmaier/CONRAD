package edu.stanford.rsl.conrad.fitting;

import java.io.Serializable;

import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.OptimizableFunction;


/**
 * Class to describe an abstract function that can be fiited to a set of 2D Points.
 * 
 * @author akmaier
 *
 */
public abstract class Function implements Serializable, OptimizableFunction{
	
	/**
	 * 
	 */
	private boolean debug = false;
	private static final long serialVersionUID = 5067981485975476424L;
	protected boolean fittingDone = false;
	protected int numberOfParameters = 0;
	public abstract double [] getParametersAsDoubleArray();
	public abstract void setParametersFromDoubleArray(double [] param);
	private double fitdatax[];
	private double fitdatay[];
	
	/**
	 * Fits the function to the given input data
	 * @param x the input data
	 * @param y the output data
	 */
	public void fitToPoints(double [] x, double [] y){
		fitdatax = x;
		fitdatay = y;
		FunctionOptimizer funcOpt = new FunctionOptimizer(this.getNumberOfParameters());
		funcOpt.setInitialX(this.getInitialX());
		funcOpt.optimizeFunction(this);
		double [] solved = funcOpt.getOptimum();
		this.setParametersFromDoubleArray(solved);
		fittingDone = true;
	};
	
	protected double[] getInitialX() {
		double init [] = new double [this.getNumberOfParameters()];
		for (int i=0; i < this.getNumberOfParameters(); i++){
			init[i] = 0;
		}
		return init;
	}
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
	
	
	@Override
	public void setNumberOfProcessingBlocks(int number) {
	}

	@Override
	public int getNumberOfProcessingBlocks() {
		// TODO Auto-generated method stub
		return 1;
	}

	@Override
	public double evaluate(double[] x, int block) {
		double value = 0;
		this.setParametersFromDoubleArray(x);
		for (int i=0; i < fitdatax.length; i++){
			value += Math.pow(this.evaluate(fitdatax[i])- fitdatay[i], 2);
		}
		value = Math.sqrt(value);
		// Catch parameterization errors and assign maximal cost.
		if (!Double.isFinite(value)) value = Double.MAX_VALUE;
		if (debug) System.out.println("Value " + value);
		return value;
	}
	
}

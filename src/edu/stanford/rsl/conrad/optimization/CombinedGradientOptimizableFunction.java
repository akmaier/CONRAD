/*
 * Copyright (C) 2015 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.optimization;

import edu.stanford.rsl.jpop.GradientOptimizableFunction;

/**
 * This class wraps an arbitrary number of jpop cost-functions into a combined cost-function value.
 * It uses Lagrange multipliers that can be set for each function individually.
 * Combining analytic gradients is also supported in case all functions provide an analytic gradient.
 * @author Martin Berger
 */
public class CombinedGradientOptimizableFunction implements
GradientOptimizableFunction {

	GradientOptimizableFunction[] stackedFunctions;
	double[] lambdas; 

	@Override
	public void setNumberOfProcessingBlocks(int number) {
		for(GradientOptimizableFunction f : stackedFunctions)
			f.setNumberOfProcessingBlocks(number);
	}

	@Override
	public int getNumberOfProcessingBlocks() {
		return stackedFunctions[0].getNumberOfProcessingBlocks();
	}

	@Override
	public double evaluate(double[] x, int block) {
		double result = 0;
		for (int i = 0; i < stackedFunctions.length; i++) {
			if(lambdas[i]!=0)
				result += lambdas[i]*stackedFunctions[i].evaluate(x, block);
		}
		return result;
	}

	@Override
	public double[] gradient(double[] x, int block) {
		double[] grads = null;
		for (int i = 0; i < this.stackedFunctions.length; i++) {
			if (lambdas[i]!=0){
				double[] res = stackedFunctions[i].gradient(x, block);
				if (res!=null){
					if(grads==null)
						grads = new double[x.length];
					for (int j = 0; j < grads.length; j++) {
						grads[j]+=lambdas[i]*res[j];
					}
				}else
					return null;
			}
		}
		return grads;
	}

	public GradientOptimizableFunction[] getStackedFunctions() {
		return stackedFunctions;
	}

	public void setStackedFunctions(
			GradientOptimizableFunction[] stackedFunctions) {
		this.stackedFunctions = stackedFunctions;
	}

	public double[] getLambdas() {
		return lambdas;
	}

	public void setLambdas(double[] lambdas) {
		this.lambdas = lambdas;
	}

}

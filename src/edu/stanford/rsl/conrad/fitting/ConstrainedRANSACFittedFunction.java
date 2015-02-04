package edu.stanford.rsl.conrad.fitting;

import java.util.Random;

/**
 * Method for RANSACing with constraints on the parameter bounds.
 * Tries that create out-of-bounds solutions are not counted.
 * 
 * @author Jiang
 *
 */
public class ConstrainedRANSACFittedFunction extends RANSACFittedFunction {

	/**
	 * Uneducational Constructor that initializes the bounds in a redandant way.
	 * Prevents the software from crashing if no bounds are set.
	 * 
	 * @param func
	 */
	public ConstrainedRANSACFittedFunction(Function func) {
		super(func);
		upperbound = new double[func.getNumberOfParameters()];
		lowerbound = new double[func.getNumberOfParameters()];
		for (int i=0;i< func.getNumberOfParameters(); i++){
			upperbound[i] = Double.POSITIVE_INFINITY;
			lowerbound[i] = Double.NEGATIVE_INFINITY;
		}
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -4078810270398436119L;
	private double [] upperbound;
	private double [] lowerbound;

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
				boolean parameterCheck = true;
				double [] param =  currentTry.getParametersAsDoubleArray();
				for (int j=0; j < currentTry.getNumberOfParameters(); j++){
					if (param[j] > upperbound[j]) parameterCheck = false;
					if (param[j] < lowerbound[j]) parameterCheck = false;
				}
				if (!parameterCheck) {
					i--;
				} else {
					double fitness = evaluateModelFittnes(x, y, currentTry );
					if (fitness > bestFit) {
						bestModel = currentTry;
						bestFit = fitness;
					}
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
			System.err.println("RANSACFittedFunction: Warining increasing epsilon to " + epsilon +"\nChoose a bigger epsilon or more tries in the future!");
			epsilon *= 1.1;
			fitToPoints(x, y);			
		}
	}

	/**
	 * @return the upperbound
	 */
	public double[] getUpperbound() {
		return upperbound;
	}

	/**
	 * @param upperbound the upperbound to set
	 */
	public void setUpperbound(double[] upperbound) {
		this.upperbound = upperbound;
	}

	/**
	 * @return the lowerbound
	 */
	public double[] getLowerbound() {
		return lowerbound;
	}

	/**
	 * @param lowerbound the lowerbound to set
	 */
	public void setLowerbound(double[] lowerbound) {
		this.lowerbound = lowerbound;
	}


}

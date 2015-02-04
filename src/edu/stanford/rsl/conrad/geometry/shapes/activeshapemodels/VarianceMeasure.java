/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels;

/**
 * Class to express a measure for the amount of principal components needed for the model to achieve the wanted variability.
 * @author Mathias Unberath
 *
 */
public class VarianceMeasure {
	/**
	 * Contains the type of the measure.
	 */
	public String type;
	
	/**
	 * Contains the value.
	 */
	public double val;
	
	/**
	 * Constructs a measure for variation and sets the type name and its value.
	 * @param name The type of variation measure.
	 * @param val The value for the measure.
	 */
	public VarianceMeasure(String name, double val){
		this.type = name;
		this.val = val;
	}
	
	/**
	 * Default constructor.
	 */
	public VarianceMeasure(){
		
	}
	
	/**
	 * Sets the type name and its value.
	 * @param name The type of variation measure.
	 * @param val The value for the measure.
	 */
	public void setMeasure(String name, double val){
		this.type = name;
		this.val = val;
	}
	
	/**
	 * Evaluates the current variation measure with respect to the eigen values, i.e. variances passed to the method.
	 * @param variation The eigenvalues to be evaluated.
	 * @return The number of principal components to use.
	 */
	public int evaluate(double[] variation){
		int dim = 0;
		if(type.toLowerCase().equals("threshold")){
			double sum = 0;
			for(int i = 0; i < variation.length; i++){
				sum += variation[i];
			}
			double[] var = new double[variation.length];
			
			var[0] = variation[0] / sum;
			for(int i = 1; i < variation.length; i++){
				var[i] = var[i-1] + variation[i] / sum;
			}
			
			while(var[dim] < val){
				dim++;
			}
		}else if(type.toLowerCase().equals("constant")){
			dim = (int)val - 1;
		}else{
			System.out.println(type + " : Unsupported measure type. Using 95% threshold instead.");
			this.type = "threshold";
			this.val = 0.95;
			dim = evaluate(variation) -1;
		}
		return dim+1;
	}

}

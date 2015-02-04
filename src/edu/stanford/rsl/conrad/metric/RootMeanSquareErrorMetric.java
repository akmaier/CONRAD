package edu.stanford.rsl.conrad.metric;

public class RootMeanSquareErrorMetric extends MeanSquareErrorMetric {

	/**
	 * 
	 */
	private static final long serialVersionUID = 332165924495213426L;

	@Override
	public double evaluate() {
		return Math.sqrt(computeMeanSquareError());
	}
	
	@Override
	public String toString() {
		return "Root Mean Square Error";
	}
	

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
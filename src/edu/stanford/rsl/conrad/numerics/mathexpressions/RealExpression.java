package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.Map;

/**
 * A wrapper class for real numbers
 * @author Rotimi X Ojo
 */
public class RealExpression extends AbstractMathExpression{
	private double val;
	
	public RealExpression(double val){
		this.val = val;
	}
	
	@Override
	public double evaluate(Map<String, Double> variablesMap){
		return val;
	}
	
	@Override
	public String toString(){
		return val+"";
	}
	
}		
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.Map;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * A wrapper class for functions
 * @author Rotimi X Ojo
 */
public class FunctionExpression extends AbstractMathExpression{
	private String function;
	private AbstractMathExpression valueExp;

	public FunctionExpression(String functionName, AbstractMathExpression valueExp){
		this.function = functionName.toLowerCase().trim();
		this.valueExp = valueExp;
	}	
	
	
	@Override
	public double evaluate(Map<String, Double> variablesMap){
		double input = valueExp.evaluate(variablesMap);
		if(function.equals("sin")){
			return Math.sin(input);
		}else if(function.equals("cos")){
			return Math.cos(input);
		}else if(function.equals("tan")){
			return Math.tan(input);
		}else if(function.equals("sind")){
			return Math.sin(General.toRadians(input));
		}else if(function.equals("cosd")){
			return Math.cos(General.toRadians(input));
		}else if(function.equals("tand")){
			return Math.tan(General.toRadians(input));
		}else if(function.equals("sqrt")){
			return Math.sqrt(input);
		}else{
			throw new UnsupportedOperationException("Function is not currently supported");
		}
	}
	@Override
	public String toString(){
		return function + "(" + valueExp.toString() + ")";
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
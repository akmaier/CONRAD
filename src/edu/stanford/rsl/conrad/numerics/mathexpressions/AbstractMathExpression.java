package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.HashMap;
import java.util.Map;

/**
 * Class to model Mathematical Expressions
 * @author Rotimi X Ojo
 */
public abstract class AbstractMathExpression {	
	
	
	public double evaluate(){
		return evaluate(new HashMap<String, Double>());
	}
	/**
	 * Determines the value of string expression represented by this class by replacing variables with values provided in map;
	 * If expression has no variable evaluate() should be called;
	 * @param variablesMap
	 * @return the evaluation result
	 */
	public abstract double evaluate(Map<String, Double> variablesMap);
	
	/**
	 * Converts a mathematical expression an equivalent string literal
	 */
	public abstract String toString();	
	

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
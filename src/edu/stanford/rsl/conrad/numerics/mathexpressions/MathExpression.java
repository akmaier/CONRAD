package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.Map;

/**
 * A wrapper class of mathematical expressions
 * @author Rotimi X Ojo
 */
public class MathExpression extends AbstractMathExpression {
	AbstractMathExpression exp;
	
	public MathExpression(String expression){
		exp = new ExpressionParser(expression).getMathExpression();
	}

	@Override
	public double evaluate(Map<String, Double> variablesMap) {
		return exp.evaluate(variablesMap);
	}

	@Override
	public String toString() {
		return exp.toString();
	}	
	

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
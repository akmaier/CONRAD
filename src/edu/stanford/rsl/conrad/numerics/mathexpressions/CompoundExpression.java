package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.Map;

/**
 * Class to model Mathematical Expressions consisting of 2 or more expressions.
 * @author Rotimi X Ojo
 */
public class CompoundExpression extends AbstractMathExpression{
	private char operator;
	private AbstractMathExpression rightExp;
	private AbstractMathExpression leftExp;
	
	public CompoundExpression( AbstractMathExpression leftExp, AbstractMathExpression rightExp,char operator){
		this.rightExp = rightExp;
		this.leftExp = leftExp;
		this.operator = operator;
	}

	@Override
	public String toString(){
		return leftExp.toString() + " " + operator + " "+ rightExp.toString();
	}
	
	@Override
	public double evaluate(Map<String,Double> varTable){
		if (operator == '=') {
			double val = rightExp.evaluate(varTable);
			varTable.put(leftExp.toString(), val);
			return val;
		}
		double leftval = leftExp.evaluate(varTable);
		double rightval = rightExp.evaluate(varTable);
		
		switch (operator) {
			case '+':return leftval + rightval;
			case '-':return leftval - rightval;
			case '*':return leftval * rightval;
			case '/':return leftval / rightval;
			default: throw new UnsupportedOperationException("Invalid Operator");
		}
		
	}
	
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
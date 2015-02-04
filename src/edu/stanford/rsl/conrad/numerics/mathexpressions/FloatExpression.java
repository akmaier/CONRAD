package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.Map;

public class FloatExpression extends AbstractMathExpression {
	String stringForm = "";
	double value = 0;

	public FloatExpression(String currToken) {
		stringForm = currToken.replace(" ", "");
		value = Double.parseDouble(stringForm);
	}

	@Override
	public double evaluate(Map<String, Double> variablesMap) {
		return value;
	}

	@Override
	public String toString() {
		return stringForm;
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
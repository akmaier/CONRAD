package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.Map;

/**
 * A wrapper class for identifiers
 * @author Rotimi X Ojo
 */
public class IdentifierExpression extends AbstractMathExpression{
	private String name;
	
	public IdentifierExpression(String name){
		this.name = name;
	}
	@Override
	public double evaluate(Map<String, Double> variablesMap){
		if(!variablesMap.containsKey(name)){
			throw new RuntimeException("Identifier is undefined");
		}
		return variablesMap.get(name);
	}
	@Override
	public String toString(){
		return name;
	}
	
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.numerics.mathexpressions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class for evaluating simple algebraic expressions
 * Custom functions can be defined in functionexpression.java
 * @author Rotimi X Ojo
 */

public class Evaluator {
	
	/**
	 * Evaluates the string literal
	 * @param expression string to be evaluated
	 * @return value of string;
	 */
	public static double getValue(String expression) {
		AbstractMathExpression exp = new MathExpression(expression);
		return exp.evaluate(new HashMap<String, Double>());
	}
	
	/**
	 * Evaluates the string literal and replaces variables with values in map;
	 * @param expression string to be evaluated
	 * @return value of string;
	 */
	public static double getValue(String expression, Map<String, Double> variablesMap) {
		AbstractMathExpression exp = new MathExpression(expression);
		return exp.evaluate(variablesMap);
	}
	
	/**
	 * Evaluates the string literals in a string vector;
	 * @param expression string vector to be evaluated
	 * @return vector value of string;
	 */
	public static SimpleVector getVectorValue(String expression) {	
		return getVectorValue(expression, new HashMap<String, Double>());
	}
	
	/**
	 * Evaluates the string literals in a string vector and replaces variables with values in map;
	 * @param expression string vector to be evaluated
	 * @return vector value of string;
	 */
	public static SimpleVector getVectorValue(String expression, Map<String, Double> variablesMap) {	
		ArrayList<Double> values = new ArrayList<Double>();
		expression = expression.trim();
		if(expression.charAt(0)=='(' && expression.charAt(expression.length()-1)==')'){
			expression = expression.substring(1,expression.length()-1);
		}		
		Scanner sc = new Scanner(expression);
		sc.useDelimiter(",");
		
		while(sc.hasNext()){
			String next = sc.next();
			values.add(getValue(next,variablesMap));
		}
		SimpleVector vector = new SimpleVector(values.size());
		Iterator<Double> it = values.iterator();
		
		int index = 0;
		while(it.hasNext()){
			vector.setElementValue(index++, it.next());
		}
		return vector;
	}
	
	/**
	 * Evaluates the string literals in a string vector;
	 * @param expression string vector to be evaluated
	 * @return point value of string;
	 */
	public static PointND getPointValue(String expression) {		
		return getPointValue(expression,new HashMap<String, Double>());
	}
	
	/**
	 * Evaluates the string literals in a string vector and replaces variables with values in map;
	 * @param expression string vector to be evaluated
	 * @return point value of string;
	 */
	public static PointND getPointValue(String expression, Map<String, Double> variablesMap) {		
		return new PointND(getVectorValue(expression,variablesMap));
	}
	
	
	/**
	 * Evaluates string literals of form (x,y,z) >(<) offset, where (x,y,z) is a normal vector;
	 * @param expression string expression of form (x,y,z) >(<) offset, where (x,y,z) is a normal vector;
	 * @return the plane defined by string expression;
	 */
	public static Plane3D getPlane(String expression) {
		return getPlane(expression, new HashMap<String, Double>());
	}
	
	public static Plane3D getPlane(String expression, Map<String, Double> variablesMap) {
		expression = expression.trim();
		if(expression.charAt(0)=='(' && expression.charAt(expression.length()-1)==')'){
			expression = expression.substring(1,expression.length()-1);
		}	

		if(expression.contains(">")){
			Plane3D plane = parsePlaneUsingDelimiter(expression, variablesMap,">");
			return plane;
		}else if(expression.contains("<")){
			Plane3D plane = parsePlaneUsingDelimiter(expression, variablesMap,"<");
			plane.flipNormal();
			return plane;
		}
		return null;
	}

	private static Plane3D parsePlaneUsingDelimiter(String expression,Map<String, Double> variablesMap, String delimiter) {
		String normexp = expression.substring(expression.indexOf(("(")), expression.indexOf(delimiter)).trim();			
		SimpleVector normal = getVectorValue(normexp,variablesMap);
		String offset = expression.substring(expression.indexOf(delimiter)+ 1);
		return new Plane3D(normal, getValue(offset,variablesMap));
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.numerics.test;



import edu.stanford.rsl.conrad.numerics.mathexpressions.Evaluator;
/**
 * Class for testing the accuracy of expressions class
 * @author Rotimi X Ojo
 *
 */

public class TestExpressions {
	
	public static void main(String [] args){
		System.out.println("Evaluating Mathematical Expressions Class:");
		int maxNum = 100;
		int x = (int)Math.random() * maxNum;
		int y = (int)Math.random() * maxNum;
		assert(Evaluator.getValue(x + "+" +y) == x + y );
		assert(Evaluator.getValue(x + "-" +y)== x - y);
		assert(Evaluator.getValue(y + "-" +x)== y - x);	
		System.out.println("\tForm [int +/- int] is accurate");
		
		x = (int)Math.random() * maxNum;
		y = (int)Math.random() * maxNum;
		assert(Evaluator.getValue("((("+x + "+" +y+")))") == x + y);
		assert(Evaluator.getValue("((("+x + "-" +y+")))")== x - y);
		assert(Evaluator.getValue("((("+y + "-" +x+")))")== y - x);
		System.out.println("\tForm [(((int +/- int)))] is accurate");
		
		double a = Math.random() * maxNum;
		double b = Math.random() * maxNum;
		assert(Evaluator.getValue("((("+a + "+" +b+")))") == a + b);
		assert(Evaluator.getValue("((("+a + "-" +b+")))")== a - b);
		assert(Evaluator.getValue("((("+b + "-" +a+")))")== b - a);
		System.out.println("\tForm [double +/- double] is accurate");

		x = 1 + (int)Math.random() * maxNum;
		y = 1 + (int)Math.random() * maxNum;
		assert(Evaluator.getValue("((("+x + "*" +y+")))") == x*y);
		assert(Evaluator.getValue("((("+x + "/" +y+")))")== x/(double)y);
		assert(Evaluator.getValue("((("+y + "/" +x+")))")== y/(double)x);
		System.out.println("\tForm [int * or / int] is accurate");
		

		a = 0.1+Math.random() * maxNum;
		b = 0.1+Math.random() * maxNum;

		assert(Evaluator.getValue("((("+a + "*" +b+")))") == a*b);
		assert(Evaluator.getValue("((("+a + "/" +b+")))")== a/b);
		assert(Evaluator.getValue("((("+b + "/" +a+")))")== b/a);
		System.out.println("\tForm [(((double * or / double)))] is accurate");
		
		double m = 0.1 + Math.random() * maxNum;
		double n = 0.1 + Math.random() * maxNum;
		double o = 1 + Math.random() * maxNum;
		assert(Evaluator.getValue("(" + m + "+" + n + ")*"+ o) == (m + n)* o);
		assert(Evaluator.getValue(o +"*(" + m + "+" + n + ")")== o*(m + n));
		assert(Evaluator.getValue(o+"/(" + m + "+" + n + ")")== o/(m+n));
		System.out.println("\tForm [(var op var) op var and var op(var op var)] is accurate");
		//
		
		m = 0.1 + Math.random() * maxNum;
		n = 0.1 + Math.random() * maxNum;
		o = 0.1 + Math.random() * maxNum;
		double p = 0.1 + Math.random() * maxNum;
		assert(Evaluator.getValue("((" + m +"+" + n+")*"+ o +")+"+ p ) == ((m + n)* o)+ p);
		assert(Evaluator.getValue("((" +m + "*(" +n + "+" + o +")))/" +p)== ((m*(n + o)))/p);
		assert(Evaluator.getValue(m +"/(("+n+"+"+o+")/" +p+")")== m/((n+o)/p));
		System.out.println("\tForm [((var op var) op var) op var and (var op(var op var)) op var] is accurate");
		
		m = 0.1 + Math.random() * maxNum;
		n = 0.1 + Math.random() * maxNum;
		o = 0.1 + Math.random() * maxNum;
		assert(Evaluator.getValue("(" +m + "+" + "sqrt("+n +"))*"+ o) == (m + Math.sqrt(n))* o);
		assert(Evaluator.getValue("sin(" +m + ")*(" +n +" + cos("+o+"))")== Math.sin(m)*(n + Math.cos(o)));
		assert(Evaluator.getValue("("+m+"+"+n +")/sqrt("+o+")")== (m + n)/Math.sqrt(o) );
		assert(Evaluator.getValue(m+"/("+n +"*tan("+o+"))") == m/(n*Math.tan(o)));
		System.out.println("\tForm [(var op func) op var and func op(func op var)] is accurate");
		
		System.out.println("Test Complete");
				
	}


}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
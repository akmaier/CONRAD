/*
 * Copyright (C) 2010-2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.weka;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.ImageJ;

/**
 * Small Example how to map trees according to the universal approximation theorem to neural nets. Epsilon is clearly greater than 0...
 * 
 * @author maier
 *
 */
public class TreeExample {

	public static void main(String [] args){
		new ImageJ();
		int wid = 30;
		int hei = 30;
		Grid2D temp = new Grid2D(wid,hei);
		NumericGridOperator op = temp.getGridOperator();
		Grid2D x1 = createBasisX(temp, 0.5, 1);
		Grid2D x2 = new Grid2D(temp);
		op.addBy(x2, 1);
		op.subtractBy(x2, x1);
		Grid2D x3 = createBasisX(temp, 0.75, 1);
		Grid2D x4 = new Grid2D(temp);
		op.addBy(x4, 1);
		op.subtractBy(x4, x3);
		Grid2D x5 = createBasisY(temp, 0.75, 1);
		Grid2D x6 = new Grid2D(temp);
		op.addBy(x6, 1);
		op.subtractBy(x6, x5);
		Grid2D ref = createReference(temp);
		ref.show("ref");
		x1.show("x1");
		x2.show("x2");
		x3.show("x3");
		x4.show("x4");
		x5.show("x5");
		x6.show("x6");
		
		SimpleMatrix mat = new SimpleMatrix(wid*hei, 6);
		SimpleVector vec1 = new SimpleVector(x1.getBuffer());
		SimpleVector vec2 = new SimpleVector(x2.getBuffer());
		SimpleVector vec3 = new SimpleVector(x3.getBuffer());
		SimpleVector vec4 = new SimpleVector(x4.getBuffer());
		SimpleVector vec5 = new SimpleVector(x5.getBuffer());
		SimpleVector vec6 = new SimpleVector(x6.getBuffer());
		mat.setColValue(0, vec1);
		mat.setColValue(1, vec2);
		mat.setColValue(1, vec3);
		mat.setColValue(3, vec4);
		mat.setColValue(2, vec5);
		mat.setColValue(5, vec6);
		SimpleVector x = new SimpleVector(ref.getBuffer());
		SimpleMatrix inv = mat.inverse(InversionType.INVERT_SVD);
		SimpleVector solution = SimpleOperators.multiply(inv, x);
		System.out.println(solution);
		//solution = new SimpleVector("[0.5; -0.25; 0.5; 0.75; 0.0; 0.0]");
		SimpleVector estimate = SimpleOperators.multiply(mat, solution);
		for (int i= 0; i <wid*hei; i++) temp.getBuffer()[i]=(float) estimate.getElement(i);
		temp.show();
	}
	
	public static Grid2D createReference(Grid2D template){
		Grid2D out = new Grid2D(template);
		for (int j=0; j < out.getHeight(); j++)
			for (int i = 0; i < out.getWidth(); i++){
				float value = 0.0f;
				if ((i<0.5*out.getWidth()) && (j < 0.75*out.getWidth())) value = 1.0f;
				if ((i>0.75*out.getWidth())) value = 1.0f;
				out.setAtIndex(i, j, value);
			}
		return out;
	}
	
	public static Grid2D createBasisX(Grid2D template, double thres, float val){
		Grid2D out = new Grid2D(template);
		for (int j=0; j < out.getHeight(); j++)
			for (int i = 0; i < out.getWidth(); i++){
				float value = 0;
				if (i < thres*out.getWidth()) value = val;
				out.setAtIndex(i, j, value);
			}
		return out;
	}
	
	public static Grid2D createBasisY(Grid2D template, double thres, float val){
		Grid2D out = new Grid2D(template);
		for (int j=0; j < out.getHeight(); j++)
			for (int i = 0; i < out.getWidth(); i++){
				float value = 0;
				if (j < thres*out.getWidth()) value = val;
				out.setAtIndex(i, j, value);
			}
		return out;
	}
	
}

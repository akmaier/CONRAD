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
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.ImageJ;
import ij.process.FloatProcessor;

/**
 * Small Example how to map trees according to the universal approximation theorem to neural nets. 
 * 
 * @author maier
 *
 */
public class TreeExample {

	public static void main(String [] args){
		new ImageJ();
		int wid = 30;
		int hei = 30;
		double sigmas [] = {0.1, 0.01, 0.001, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 0.5, 0.25, 32};
		Grid2D temp = new Grid2D(wid,hei);
		NumericGridOperator op = temp.getGridOperator();
		
		double [] xes = {0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.73, 0.75, 0.77, 0.8, 0.9, 1};
		double [] yes = {0, 0.1, 0.2, 0.22, 0.25, 0.27, 0.3, 0.4, 0.5, 0.6, 0.73, 0.75, 0.77, 0.8, 0.9, 1};
		Grid2D [] basis = new Grid2D[(xes.length * 2) + (yes.length*2)];
		for (int i=0; i < xes.length; i++){
			basis[i*2] = createBasisX(temp, xes[i], 1);
			basis[i*2+1] = new Grid2D(temp);
			op.addBy(basis[i*2+1], 1);
			op.subtractBy(basis[i*2+1], basis[i*2]);		
		}
		for (int i=xes.length; i < yes.length + (xes.length); i++){
			basis[i*2] = createBasisY(temp, yes[i-xes.length], 1);
			basis[i*2+1] = new Grid2D(temp);
			op.addBy(basis[i*2+1], 1);
			op.subtractBy(basis[i*2+1], basis[i*2]);		
		}
		
		Grid2D ref = createReference(temp);
		ref.show("ref");
		Grid2D [] blurs = new Grid2D [(basis.length) * (sigmas.length+1)];
		for (int i=0; i< basis.length; i++){
			blurs[i*(sigmas.length +1)] = basis[i];
			for (int j =0; j < sigmas.length; j++){
				blurs[(i*(sigmas.length+1))+j+1] = blur(basis[i], sigmas[j]);
			}
		}

		SimpleMatrix mat = new SimpleMatrix(wid*hei, blurs.length);
		for (int i = 0; i < blurs.length; i++) {
			SimpleVector vec = new SimpleVector(blurs[i].getBuffer());
			mat.setColValue(i, vec);
		}
		
		SimpleVector x = new SimpleVector(ref.getBuffer());
		SimpleMatrix inv = mat.inverse(InversionType.INVERT_SVD);
		SimpleVector solution = SimpleOperators.multiply(inv, x);
		System.out.println(solution);
		//solution = new SimpleVector("[0.5; -0.25; 0.5; 0.75; 0.0; 0.0]");
		SimpleVector estimate = SimpleOperators.multiply(mat, solution);
		for (int i= 0; i <wid*hei; i++) temp.getBuffer()[i]=(float) estimate.getElement(i);
		new Grid2D(temp).show();
		op.subtractBy(temp, ref);
		op.abs(temp);
		System.out.println("epsilon = " + op.max(temp) + " " +op.sum(temp));
	}

	public static Grid2D blur(Grid2D in, double sigma){
		FloatProcessor img = ImageUtil.wrapGrid2D(new Grid2D(in));
		img.blurGaussian(sigma);
		return ImageUtil.wrapFloatProcessor(img);
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

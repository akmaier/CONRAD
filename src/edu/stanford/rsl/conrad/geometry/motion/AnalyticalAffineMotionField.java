package edu.stanford.rsl.conrad.geometry.motion;


import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.DoubleFunction;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;





/**
 * MotionField to handle affine rotation and affine modulated translational motion.
 * The motion paramters, i.e. angle, axis, translation, can be given as lambda expressions or method references
 * 
 * @author berger
 *
 */
public abstract class AnalyticalAffineMotionField extends AbstractAffineMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7314670417660902689L;

	protected SimpleMatrix preTransform;
	protected SimpleMatrix postTransform;

	public void setPostTransform(SimpleMatrix postTransform) {
		this.postTransform = postTransform;
	}

	public SimpleMatrix getPostTransform() {
		return postTransform;
	}

	public void setPreTransform(SimpleMatrix preTransform) {
		this.preTransform = preTransform;
	}

	public SimpleMatrix getPreTransform() {
		return preTransform;
	}

	@Override
	public Transform getTransform(double initialTime, double time) {
		SimpleMatrix variableTransform = getAffineMatrix(initialTime, time);
		if(preTransform!=null)
			variableTransform=SimpleOperators.multiplyMatrixProd(variableTransform, preTransform);
		if(postTransform!=null)
			variableTransform=SimpleOperators.multiplyMatrixProd(postTransform, variableTransform);
		int rows = variableTransform.getRows();
		int cols = variableTransform.getCols();
		variableTransform.divideBy(variableTransform.getElement(rows-1, cols-1));
		return new AffineTransform(variableTransform.getSubMatrix(rows-1, cols-1),variableTransform.getSubCol(0, cols-1, rows-1));
	}

	public abstract SimpleMatrix getAffineMatrix(double initialTime, double time);

	protected SimpleVector evaluateFunctionArray(DoubleFunction[] array, double arg){
		SimpleVector out = new SimpleVector(array.length); 
		for (int i = 0; i < array.length; i++) {
			out.setElementValue(i, array[i].f(arg));
		}
		return out;
	}





}
/*
 * Copyright (C) 2010-2014 Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
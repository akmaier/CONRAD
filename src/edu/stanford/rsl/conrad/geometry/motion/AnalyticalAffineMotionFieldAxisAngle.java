package edu.stanford.rsl.conrad.geometry.motion;


import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
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
public class AnalyticalAffineMotionFieldAxisAngle extends AnalyticalAffineMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2402284037682473196L;
	
	/**
	 * Creates a new rotational and translation MotionField. 
	 * It transforms points with a rotation around the given axis and center and translates them afterwards. 
	 * The translation is an affine motion field as well which gives further flexibility.
	 * The angle is given in radians.
	 *  
	 * @param transformationCenter
	 * @param axis
	 * @param angle
	 */

	public AnalyticalAffineMotionFieldAxisAngle() {
	}
	
	DoubleFunction[] transformationCenter;
	DoubleFunction[] axis;
	DoubleFunction angle;
	DoubleFunction[] translation;


	public AnalyticalAffineMotionFieldAxisAngle (DoubleFunction[] transformationCenter, DoubleFunction[] axis, DoubleFunction angle, DoubleFunction[] translation){
		this.transformationCenter = transformationCenter;
		this.translation = translation;
		this.axis = axis;
		this.angle = angle;
	}

	

	/**
	 * Returns the interpolated transform between initial time and time.
	 * Here the lambda expressions are evaluated and the current transform is build
	 * 
	 * @param initialTime
	 * @param time
	 * @return the transform
	 * @throws Exception 
	 */
	@Override
	public SimpleMatrix getAffineMatrix(double initialTime, double time) {

		// first get the current rotation center
		SimpleVector transformCenter = evaluateFunctionArray(transformationCenter,time);

		// now we determine the current rotation axis and the angle
		SimpleVector curAxis = evaluateFunctionArray(axis, time);
		double curAngle = angle.f(time);

		// finally we evaluate the current translation
		SimpleVector curTranslation = evaluateFunctionArray(translation, time);

		SimpleMatrix out = new SimpleMatrix(curAxis.getLen()+1,curAxis.getLen()+1);
		out.setElementValue(out.getRows()-1, out.getCols()-1, 1);
		
		SimpleMatrix rot = Rotations.createRotationMatrixAboutAxis(new Axis(curAxis), curAngle);
		out.setSubMatrixValue(0, 0, rot);	
		
		// Translation is build by: currentTranslation + transformCenter - Rotation*transformCenter
		// Computed directly to avoid matrix multiplications!
		// Validity is easily tested by plugging i the transform Center, as the result should be the transform center
		curTranslation.add(transformCenter);
		curTranslation.subtract(SimpleOperators.multiply(rot, transformCenter));
		out.setSubColValue(0, out.getCols()-1, curTranslation);
		return out;
	}
	
	public DoubleFunction[] getTransformationCenter() {
		return transformationCenter;
	}



	public void setTransformationCenter(DoubleFunction[] transformationCenter) {
		this.transformationCenter = transformationCenter;
	}



	public DoubleFunction[] getAxis() {
		return axis;
	}



	public void setAxis(DoubleFunction[] axis) {
		this.axis = axis;
	}



	public DoubleFunction getAngle() {
		return angle;
	}



	public void setAngle(DoubleFunction angle) {
		this.angle = angle;
	}



	public DoubleFunction[] getTranslation() {
		return translation;
	}



	public void setTranslation(DoubleFunction[] translation) {
		this.translation = translation;
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
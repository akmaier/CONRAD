package edu.stanford.rsl.conrad.geometry.motion;

import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * MotionField to handle rotational motion.
 * 
 * @author akmaier
 *
 */
public class RotationMotionField extends SimpleMotionField {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -5199935079209350635L;
	protected Translation toCenter;
	protected Translation back;
	protected SimpleVector axis;
	protected double angle;
	protected PointND center;
	
	/**
	 * Creates a new rotational MotionField. It transforms points with a rotation around the given axis and center point transformationCenter. The angle is given in radians. 
	 * @param transformationCenter
	 * @param axis
	 * @param angle
	 */
	public RotationMotionField (PointND transformationCenter, SimpleVector axis, double angle){
		back = new Translation(transformationCenter.getAbstractVector());
		toCenter = back.inverse();
		warp = new IdentityTimeWarper();
		center = transformationCenter;
		this.axis = axis;
		this.angle = angle;
	}

	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		Transform rotation = getTransform(initialTime, time);
		PointND p = back.transform(rotation.transform(toCenter.transform(initialPosition)));
		return p;
	}
	
	/**
	 * Returns the interpolated transform between initial time and time.
	 * @param initialTime
	 * @param time
	 * @return the transform
	 */
	public Transform getTransform(double initialTime, double time){
		Transform rotation = new ScaleRotate(Rotations.createRotationMatrixAboutAxis(new Axis(axis), angle * (warp.warpTime(time)-warp.warpTime(initialTime))));
		return rotation;
	}

	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
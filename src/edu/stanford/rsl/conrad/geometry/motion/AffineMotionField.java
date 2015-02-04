package edu.stanford.rsl.conrad.geometry.motion;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import Jama.EigenvalueDecomposition;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * MotionField to handle affine rotation and translational motion.
 * 
 * @author berger
 *
 */
public class AffineMotionField extends AbstractAffineMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6748536919627679826L;
	protected Translation toCenter;
	protected Translation back;
	protected SimpleVector axis;
	protected double angle;
	protected PointND center;
	protected SimpleVector translation;
	protected SimpleMatrix rotationMat;

	/**
	 * Creates a new rotational and translation MotionField. 
	 * It transforms points with a rotation around the given axis and center and translates them afterwards. 
	 * The angle is given in radians.
	 *  
	 * @param transformationCenter
	 * @param axis
	 * @param angle
	 */
	public AffineMotionField (PointND transformationCenter, SimpleVector axis, double angle, SimpleVector translation){
		back = new Translation(transformationCenter.getAbstractVector());
		toCenter = back.inverse();
		warp = new IdentityTimeWarper();
		center = transformationCenter;
		this.axis = axis;
		this.angle = angle;
		this.rotationMat = null;
		this.translation = translation;
	}
	
	/**
	 * Creates a new rotation motion field.
	 * It transforms points with a rotation around the given axis and center and translates them afterwards. 
	 * The angle is given in radians.
	 *  
	 * @param transformationCenter
	 * @param axis
	 * @param angle
	 */
	public AffineMotionField (PointND transformationCenter, SimpleVector axis, double angle){
		this(transformationCenter, axis, angle, new SimpleVector(transformationCenter.getDimension()));
	}

	/**
	 * Creates a new rotational and translation MotionField. 
	 *  
	 * @param transformationCenter the center point of the rotation
	 * @param rotMat the rotation matrix
	 * @param translation the translation
	 */
	public AffineMotionField (PointND transformationCenter, SimpleMatrix rotMat, SimpleVector translation){
		back = new Translation(transformationCenter.getAbstractVector());
		toCenter = back.inverse();
		warp = new IdentityTimeWarper();
		center = transformationCenter;
		this.axis = null;
		this.angle = 0;
		this.rotationMat = rotMat;
		this.translation = translation;
	}

	
	/**
	 * Returns the interpolated transform between initial time and time.
	 * @param initialTime
	 * @param time
	 * @return the transform
	 * @throws Exception 
	 */
	@Override
	public Transform getTransform(double initialTime, double time) {

		Transform affine = null;
		if (axis != null){
			affine = new AffineTransform(
					Rotations.createRotationMatrixAboutAxis(new Axis(axis), angle * (warp.warpTime(time)-warp.warpTime(initialTime))),
					translation.multipliedBy(warp.warpTime(time)-warp.warpTime(initialTime))
					);
		}
		else{
			// TODO: add support for rotation matrices (Use axis/angle representation)
			throw new NotImplementedException();
		}
		affine = new ComboTransform(toCenter,affine,back);
		return affine;
	}



}
/*
 * Copyright (C) 2010-2014 Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
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
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * MotionField to handle affine rotation and translational motion.
 * 
 * @author berger
 *
 */
public abstract class AbstractAffineMotionField extends SimpleMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2996698945609143759L;

	/**
	 * {@inheritDoc}
	 */
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		return this.getTransform(initialTime, time).transform(initialPosition);
	}

	/**
	 * Returns the interpolated transform between initial time and time.
	 * @param initialTime
	 * @param time
	 * @return the transform
	 * @throws Exception 
	 */
	public abstract Transform getTransform(double initialTime, double time);


	public SimpleMatrix getTransformAsMatrix(double initialTime, double time){
		Transform tform = getTransform(initialTime, time);
		SimpleMatrix mat = new SimpleMatrix(4,4);
		mat.setSubMatrixValue(0, 0, tform.getRotation(3));
		mat.setSubColValue(0, 3, tform.getTranslation(3));
		mat.setElementValue(3, 3, 1);
		return mat;
	}
}
/*
 * Copyright (C) 2010-2014 Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
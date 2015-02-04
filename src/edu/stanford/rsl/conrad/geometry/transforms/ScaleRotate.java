package edu.stanford.rsl.conrad.geometry.transforms;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This is a decorator class for a scale-rotation matrix. it performs a scale-rotate transformation on point and direction vectors.
 * @author Rotimi X Ojo
 *
 */
public class ScaleRotate extends Transform{

	private static final long serialVersionUID = 4514828466172949672L;
	private SimpleMatrix scaleRotate = null;
	
	/**
	 * Initialize a scale-rotation transform using a rotation matrix
	 * @param t is a scale-rotation matrix
	 */
	public ScaleRotate(SimpleMatrix t){
		scaleRotate = t;
	}

	/**
	 * Applies scale-rotate transformation on a given point
	 * @param point is point to be scale rotated
	 * @return point y = Ax, where A is the scale-rotate matrix wrapped by this class and x is the point to be rotated.
	 */
	@Override
	public PointND transform(PointND point) {
		return new PointND(SimpleOperators.multiply(scaleRotate, point.getAbstractVector()));
	}

	/**
	 * Applies scale-rotate transformation on a given direction
	 * @param dir is direction to be scale rotated.
	 * @return vector y = Ax, where A is the scale-rotate matrix wrapped by this class and x is the vector to be rotated.
	 */
	@Override
	public SimpleVector transform(SimpleVector dir) {
		return SimpleOperators.multiply(scaleRotate, dir);
	}

	@Override
	public Transform inverse() {
		return new ScaleRotate(scaleRotate.inverse(SimpleMatrix.InversionType.INVERT_QR));		
	}


	@Override
	public Transform clone() {
		return new ScaleRotate(scaleRotate.clone());
	}


	@Override
	public SimpleMatrix getData() {
		return scaleRotate;
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
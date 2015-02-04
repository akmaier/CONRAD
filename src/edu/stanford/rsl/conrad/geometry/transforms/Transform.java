/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.transforms;

import java.io.Serializable;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


public abstract class Transform implements Serializable {


	private static final long serialVersionUID = -5051528144142521617L;
	
	/**
	 * Transforms the given point and returns a new transformed point.
	 * @param point is point to be tranformed
	 * @return is resultant point
	 */
	public abstract PointND transform(PointND point);
	
	/**
	 * Transforms the given vector
	 * @param dir is vector to be transformed
	 * @return is resultant vector
	 */
	public abstract SimpleVector transform(SimpleVector dir);
	
	public abstract Transform inverse();
	
	public abstract Transform clone();
	
	public abstract Object getData();
	
	public SimpleMatrix getRotation(int dimensions){
		SimpleMatrix rotation = new SimpleMatrix(dimensions, dimensions);
		for (int i =0 ; i<dimensions; i++){
			SimpleVector e = new SimpleVector(dimensions);
			e.setElementValue(i, 1.0);
			rotation.setColValue(i, transform(e));
		}
		return rotation;
	}
	
	public SimpleVector getTranslation(int dimensions){
		SimpleVector e = new SimpleVector(dimensions);
		PointND origin = new PointND(e);
		SimpleVector transformedPoint = transform(origin).getAbstractVector();
		return transformedPoint;
	}

}
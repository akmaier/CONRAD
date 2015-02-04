package edu.stanford.rsl.conrad.geometry.motion;

import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class DualMotionField extends SimpleMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -202437787576272143L;
	protected HalfSpaceBoundingCondition halfSpaceOne;
	protected MotionField one;
	protected MotionField two;
	
	public DualMotionField (HalfSpaceBoundingCondition halfSpaceOne, MotionField one, MotionField two){
		this.one = one;
		this.two = two;
		this.halfSpaceOne = halfSpaceOne;
	}
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		HalfSpaceBoundingCondition currentHS = new HalfSpaceBoundingCondition(new Plane3D(one.getPosition(halfSpaceOne.getBoundingSurface().getPoint(), 0, initialTime), halfSpaceOne.getBoundingSurface().getNormal()));
		if (currentHS.isSatisfiedBy(initialPosition)){
			return one.getPosition(initialPosition, initialTime, time);
		} else {
			return two.getPosition(initialPosition, initialTime, time);
		}
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
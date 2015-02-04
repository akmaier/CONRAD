package edu.stanford.rsl.conrad.geometry.bounds;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * This class describes the bounds defined by a {@link PointND}. This class is useful when a plane describes the extremum of an arbitrary shape.
 * @author Rotimi X Ojo
 * @author Akm
 */
public class HalfSpaceBoundingCondition extends AbstractBoundingCondition {

	private static final long serialVersionUID = -1161793937985226484L;
	protected Plane3D plane;
	private boolean isFlipped = false;
	
	/**
	 * Initialize new  HalfSpace Bounding Condition with a plane.
	 * @param plane is plane describing the extremum of an arbitrary shape.
	 */
	public HalfSpaceBoundingCondition(Plane3D plane){
		this.plane = plane;
	}
	
	/**
	 * Initialize new HalfSpace Bounding Condition with a plane defined by two points.
	 * @param one is first point on plane describing the extremum of an arbitrary shape.
	 * @param two is second point on plane describing the extremum of an arbitrary shape.
	 */
	public HalfSpaceBoundingCondition(PointND one, PointND two){
		this.plane = new Plane3D(one, SimpleOperators.subtract(two.getAbstractVector(), one.getAbstractVector()), new SimpleVector(0, 0, 1));
	}
	
	public HalfSpaceBoundingCondition(HalfSpaceBoundingCondition hsbc){
		super(hsbc);
		plane = (hsbc.plane!=null) ? new Plane3D(hsbc.plane) : null;
		isFlipped = hsbc.isFlipped;
	}

	@Override
	public boolean isSatisfiedBy(PointND point) {
		boolean val = (plane.computeDistance(point) > -CONRAD.FLOAT_EPSILON);
		if(isFlipped){
			return val == true? false: true;
		}			
		return val;
	}	


	@Override
	public Plane3D getBoundingSurface() {
		return plane;
	}

	@Override
	public void flipCondition() {
		isFlipped = isFlipped==true ? false:true;			
	}

	@Override
	public AbstractBoundingCondition clone() {
		return new HalfSpaceBoundingCondition(this);
	}
	
}
/*
 * Copyright (C) 2010 - Rotimi X Ojo, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
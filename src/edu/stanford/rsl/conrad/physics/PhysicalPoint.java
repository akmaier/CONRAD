package edu.stanford.rsl.conrad.physics;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SortablePoint;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class PhysicalPoint extends SortablePoint {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1259588295137355141L;
	private PhysicalObject object;
	protected double hitOrientation;	// used to determine from which direction a triangle has hit the ray

	public PhysicalPoint(PointND p) {
		super(p);
	}

	public PhysicalPoint(SimpleVector add) {
		super(add);
	}

	public PhysicalPoint(double ... d) {
		super(d);
	}

	/**
	 * @param object the object to set
	 */
	public void setObject(PhysicalObject object) {
		this.object = object;
	}

	/**
	 * @return the object
	 */
	public PhysicalObject getObject() {
		return object;
	}
	
	
	public double getHitOrientation() {
		return hitOrientation;
	}

	
	public void setHitOrientation(double hitOrientation) {
		this.hitOrientation = hitOrientation;
	}

	
	public boolean equals(PhysicalPoint p){
		return super.equals(p) && (object.equals(p.object));
	}	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
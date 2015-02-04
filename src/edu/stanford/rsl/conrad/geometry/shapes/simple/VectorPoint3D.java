package edu.stanford.rsl.conrad.geometry.shapes.simple;

import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to model a 3D point which contains a vector value.  Point can be used to model a vector in space or a colored point in space.
 * 
 * @author akmaier
 *
 */
public class VectorPoint3D extends Point3D {

	private static final long serialVersionUID = 3854685720292206922L;
	protected SimpleVector vector;

	/**
	 * Creates a new VectorPoint3D at coordinates (x, y, z).
	 * @param x the x coordinate
	 * @param y the y coordinate
	 * @param z the z coordinate
	 * @param vector the vector as list of double values
	 */
	public VectorPoint3D(double x, double y, double z, double... vector){
		this(x, y, z, new SimpleVector(vector));
	}
	
	/**
	 * Creates a new VectorPoint3D at coordinates (x, y, z).
	 * @param x the x coordinate
	 * @param y the y coordinate
	 * @param z the z coordinate
	 * @param vector the vector as SimpleVector
	 */
	public VectorPoint3D(double x, double y, double z, SimpleVector vector){
		super(x, y, z);
		this.vector = new SimpleVector(vector);
	}
	
	/**
	 * Creates a new VectorPoint3D at point.
	 * @param point the point
	 * @param vector the vector as SimpleVector
	 */
	public VectorPoint3D(PointND point, SimpleVector vector){
		super(point);
		this.vector = new SimpleVector(vector);
	}
	
	/**
	 * Creates a new VectorPoint3D at point.
	 * @param point the point
	 * @param vector the vector as list of double values
	 */
	public VectorPoint3D(PointND point, double ... vector) {
		this(point, new SimpleVector(vector));
	}

	/**
	 * @return the vector
	 */
	public SimpleVector getVector() {
		return vector;
	}

	/**
	 * @param vector the vector to set
	 */
	public void setVector(SimpleVector vector) {
		this.vector = vector;
	}
	

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
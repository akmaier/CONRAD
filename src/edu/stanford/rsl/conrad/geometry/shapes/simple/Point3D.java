package edu.stanford.rsl.conrad.geometry.shapes.simple;



/**
 * Wrapper class to model a 3D point.
 * 
 * @author akmaier
 *
 */
public class Point3D extends PointND {


	/**
	 * 
	 */
	private static final long serialVersionUID = -3570269971680307757L;
	
	/**
	 * Creates a new 3D Point from a list of coordinates. Asserts that the list id of length 3.
	 * @param coordinates
	 */
	public Point3D(double ... coordinates){
		super(coordinates);
		assert (coordinates.length == 3);
	}

	/**
	 * Copy constructor. Asserts that the dimension is 3.
	 * @param point
	 */
	public Point3D(PointND point) {
		super(point);
		assert(point.getDimension() == 3);
	}

	/**
	 * returns the x coordinate
	 * @return x
	 */
	public double getX(){
		return coordinates.getElement(0);
	}
	
	/**
	 * returns the y coordinate
	 * @return y
	 */
	public double getY(){
		return coordinates.getElement(1);
	}
	
	/**
	 * returns the z coordinate
	 * @return z
	 */
	public double getZ(){
		return coordinates.getElement(2);
	}
	
	/**
	 * sets the x coordinate
	 * @param x coordinate
	 */
	public void setX(double x){
		coordinates.setElementValue(0, x);
	}
	
	/**
	 * sets the y coordinate
	 * @param y coordinate
	 */
	public void setY(double y){
		coordinates.setElementValue(1, y);
	}
	
	/**
	 * sets the z coordinate
	 * @param z coordinate
	 */
	public void setZ(double z){
		coordinates.setElementValue(2, z);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
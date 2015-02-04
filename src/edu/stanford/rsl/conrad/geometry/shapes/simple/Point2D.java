package edu.stanford.rsl.conrad.geometry.shapes.simple;


/**
 * Wrapper class to model a 2D point.
 * 
 * @author akmaier
 *
 */
public class Point2D extends PointND {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5759702343444016617L;

	public Point2D(double ... coordinates){
		super(coordinates);
		assert (coordinates.length == 2);
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
		coordinates.setElementValue(0, y);
	}
	

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
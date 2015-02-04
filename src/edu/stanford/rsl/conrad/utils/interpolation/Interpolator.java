package edu.stanford.rsl.conrad.utils.interpolation;

import java.io.Serializable;

/**
 * This is a abstract class for interpolators.
 * Interpolation is a method of constructing new data points within the range of a discrete set of known data points.
 * For more on interpolation see http://en.wikipedia.org/wiki/Interpolation
 * @author Rotimi X Ojo
 *
 */
public abstract class Interpolator implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 5165462540768302312L;
	protected  double xFloor;
	protected  double xCeiling;
	protected  double yFloor;
	protected  double yCeiling;

	public void setXPoints(double floor, double ceiling){
		this.xFloor = floor;
		this.xCeiling = ceiling;
	}
	
	public void setYPoints(double floor, double ceiling){
		this.yFloor = floor;
		this.yCeiling = ceiling;
	}
	
	public abstract double InterpolateYValue(double key);
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
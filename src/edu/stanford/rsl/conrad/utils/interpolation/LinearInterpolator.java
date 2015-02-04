package edu.stanford.rsl.conrad.utils.interpolation;



/**
 * This is a class for interpolating between two data points using a linear polynomial.
 * Interpolation is a method of constructing new data points within the range of a discrete set of known data points.
 * For more on linear interpolation see http://en.wikipedia.org/wiki/Linear_interpolation
 * @author Rotimi X Ojo
 *
 */
public class LinearInterpolator extends Interpolator {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4895659448323575250L;

	public double InterpolateYValue(double key){
		return yFloor + ((key - xFloor)*(yCeiling - yFloor)/(xCeiling - xFloor));
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
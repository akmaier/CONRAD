package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import java.io.Serializable;

/**
 * TimeWarper is a class to warp time to match certain periodic, quasi-periodic, and non periodic behavior.
 * 
 * @author akmaier
 *
 */
public abstract class TimeWarper implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6032593933715940914L;

	/**
	 * Applies the time warping to the given input time
	 * @param time the input time
	 * @return the output time
	 */
	public abstract double warpTime(double time);
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
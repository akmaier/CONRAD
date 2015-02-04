package edu.stanford.rsl.conrad.geometry.motion.timewarp;



/**
 * Models a non periodic time constraint. Input time is returned as output time.
 * @author akmaier
 *
 */
public class IdentityTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6668711012221834486L;

	@Override
	public double warpTime(double time) {
		return time;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
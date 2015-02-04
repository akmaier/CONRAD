package edu.stanford.rsl.conrad.geometry.motion.timewarp;



/**
 * Models a non periodic time constraint. Input time is returned as output time plus scaling.
 * @author berger
 *
 */
public class ScaledIdentitiyTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8494453378023449283L;
	
	private double maxVal = 1;
	private double minVal = 0;

	public ScaledIdentitiyTimeWarper(double min, double max){
		maxVal = max;
		minVal = min;
	}
	
	@Override
	public double warpTime(double time) {
		return time*(maxVal-minVal)+minVal;
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
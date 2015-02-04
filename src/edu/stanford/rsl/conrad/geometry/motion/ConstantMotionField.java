package edu.stanford.rsl.conrad.geometry.motion;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class ConstantMotionField extends SimpleMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6413969556338466583L;

	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		return initialPosition;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
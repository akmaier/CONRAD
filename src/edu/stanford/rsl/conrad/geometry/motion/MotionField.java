/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.motion;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public interface MotionField {

	/**
	 * Determines the position at a given initialPosition and a given time where time = 0  is the initial position and time = 1 is the end position.
	 * @param initialPosition
	 * @param initialTime
	 * @param time
	 * @return the position at the time
	 */
	public PointND getPosition(PointND initialPosition, double initialTime, double time);
	
	public void setTimeWarper(TimeWarper warp);
	
	public TimeWarper getTimeWarper();
	
	public ArrayList<PointND> getPositions(PointND initialPosition, double initialTime, double ... times);
	
	/**
	 * implements a position look up from one time to another for many points
	 * @param initialTime the inital time
	 * @param time the end time
	 * @param initialPositions the initial positions
	 * @return the positions as array list
	 */
	public ArrayList<PointND> getPositions(double initialTime, double time, PointND ... initialPositions);
	
}

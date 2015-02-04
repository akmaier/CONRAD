/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.motion;

import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;


public abstract class SimpleMotionField implements MotionField, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -352375092478769290L;
	protected TimeWarper warp;
	
	public abstract PointND getPosition(PointND initialPosition, double initialTime, double time);

	public ArrayList<PointND> getPositions(PointND initialPosition, double initialTime,
			double ... times) {
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i=0; i< times.length; i++){
			list.add(getPosition(initialPosition, initialTime, times[i]));
		}
		return list;
	}
	
	public ArrayList<PointND> getPositions(double initialTime,
			double time, PointND ... initialPositions) {
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i=0; i< initialPositions.length; i++){
			list.add(getPosition(initialPositions[i], initialTime, time));
		}
		return list;
	}

	public TimeWarper getTimeWarper() {
		return warp;
	}

	public void setTimeWarper(TimeWarper warp) {
		this.warp = warp;
	}

}

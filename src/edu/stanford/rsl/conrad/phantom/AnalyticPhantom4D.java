package edu.stanford.rsl.conrad.phantom;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;

public abstract class AnalyticPhantom4D extends AnalyticPhantom implements
		MotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5680252649645704710L;
	protected TimeWarper warper;
	
	@Override
	public void setTimeWarper(TimeWarper warp) {
		warper = warp;
	}

	@Override
	public TimeWarper getTimeWarper() {
		return warper;
	}
	
	public MotionField getMotionField(){
		return this;
	}
	
	@Override
	public ArrayList<PointND> getPositions(double initialTime,
			double time, PointND ... initialPositions) {
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i=0; i< initialPositions.length; i++){
			list.add(getPosition(initialPositions[i], initialTime, time));
		}
		return list;
	}
	
	/**
	 * Creates the scene at time t given the sampling factors. Sampling factors are only used if tessellation is required.
	 * @param voxelSizeX
	 * @param voxelSizeY
	 * @param voxelSizeZ
	 * @param samplingU
	 * @param samplingV
	 * @param time
	 * @return the scene
	 */
	public abstract PrioritizableScene getScene(double time);

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
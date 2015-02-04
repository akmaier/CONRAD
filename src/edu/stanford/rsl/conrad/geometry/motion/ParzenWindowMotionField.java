/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.motion;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


/**
 * This is an abstract class to describe a motion field that uses a Parzen window for interpolation. 
 * The class tessellates the surface at the two time points and uses these points to interpolate the motion vectors. 
 * Due to the high computational effort of tessellation on the CPU we store intermediate tessellation results for later use. 
 * Interpolation is done via a Parzen window.
 * @author akmaier
 *
 */

public abstract class ParzenWindowMotionField implements MotionField,
Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6548767514419099244L;
	HashMap<Double, PointND[]> timePointMap;
	/**
	 * Standard Deviation of the Gaussian Window in world coordinate metric, i.e. [mm] for XCat.
	 */
	double sigma;
	TimeWarper warp = new IdentityTimeWarper();

	abstract PointND [] getRasterPoints (double time);

	/**
	 * Contructor
	 * @param sigma the sigma [mm]
	 */
	public ParzenWindowMotionField(double sigma) {
		this.sigma = sigma;
		timePointMap = new HashMap<Double, PointND[]>();
	}




	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		SimpleVector summation = new SimpleVector(0,0,0);
		float weightsum = 0;
		PointND[] from = getRasterPoints(initialTime);
		PointND[] to = getRasterPoints(time);
		double acc = 70; // Highest number before summations yield NaN
		for (int i=0; i< from.length; i++){
			float weight = (float) Math.exp((-0.5*Math.pow(from[i].euclideanDistance(initialPosition),2)/Math.pow(sigma,2))+acc);

			weightsum += weight;
			SimpleVector direction = new SimpleVector(to[i].getAbstractVector());
			direction.subtract(from[i].getAbstractVector());
			summation.add(direction.multipliedBy(weight));
		}
		if(Math.abs(weightsum) < 0.00000001) {
			summation.multiplyBy(0);
		} else {
			summation.divideBy(weightsum);
		}
		summation.add(initialPosition.getAbstractVector());
		return new PointND(summation);
	}

	@Override
	public void setTimeWarper(TimeWarper warp) {
		this.warp= warp;
	}

	@Override
	public TimeWarper getTimeWarper() {
		return warp;
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		ArrayList<PointND> result = new ArrayList<PointND>();
		for (int i =0; i< times.length; i++){
			result.add(getPosition(initialPosition, initialTime, times[i]));
		}
		return result;
	}

	@Override
	public ArrayList<PointND> getPositions(double initialTime,
			double time, PointND ... initialPositions) {
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int j=0; j< initialPositions.length; j++){
			SimpleVector summation = new SimpleVector(0,0,0);
			float weightsum = 0;
			PointND[] from = getRasterPoints(initialTime);
			PointND[] to = getRasterPoints(time);
			double acc = 70; // Highest number before summations yield NaN
			for (int i=0; i< from.length; i++){
				float weight = (float) Math.exp((-0.5*Math.pow(from[i].euclideanDistance(initialPositions[j]),2)/Math.pow(sigma,2))+acc);
				weightsum += weight;
				SimpleVector direction = new SimpleVector(to[i].getAbstractVector());
				direction.subtract(from[i].getAbstractVector());
				summation.add(direction.multipliedBy(weight));
			}
			if(Math.abs(weightsum) < 0.00000001) {
				summation.multiplyBy(0);
			} else {
				summation.divideBy(weightsum);
			}
			summation.add(initialPositions[j].getAbstractVector());
			PointND newPoint = new PointND(summation);
			list.add(newPoint);
		}
		return list;
	}

}

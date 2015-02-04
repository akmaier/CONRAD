package edu.stanford.rsl.conrad.geometry.motion;

import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


/**
 * Implements a linear interpolating motion field based on points over time. 
 * 
 * @author akmaier
 *
 */
public class PointBasedMotionField implements MotionField, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1286210731201948669L;
	protected TimeWarper warp;
	protected ArrayList<ArrayList<PointND>> timePoints = new ArrayList<ArrayList<PointND>>();
	private int context;
	protected int timeSamplePoints = 10;

	public PointBasedMotionField(TimeVariantSurfaceBSpline [] variants, int context){
		this.context = context;
		warp = new IdentityTimeWarper();
		for (int i = 0; i < variants.length; i++){
			if (variants[i].getNumberOfTimePoints() > timeSamplePoints) timeSamplePoints = variants[i].getNumberOfTimePoints();
		}		
		for (int t = 0; t < timeSamplePoints; t++) {
			ArrayList<PointND> list = new ArrayList<PointND>();
			for (int i = 0; i < variants.length; i++){
				list.addAll(variants[i].getControlPoints(t));
			}
			timePoints.add(list);
		}
	}

	public PointBasedMotionField(ArrayList<TimeVariantSurfaceBSpline> variants, int context){
		this.context = context;
		warp = new IdentityTimeWarper();
		for (int i = 0; i < variants.size(); i++){
			if (variants.get(i).getNumberOfTimePoints() > timeSamplePoints) timeSamplePoints = variants.get(i).getNumberOfTimePoints();
		}
		for (int t = 0; t <timeSamplePoints; t++) {
			timePoints.add(new ArrayList<PointND>());
			for (int i = 0; i < variants.size(); i++){
				timePoints.get(t).addAll(variants.get(i).getControlPoints(t));
			}
		}
	}

	public PointBasedMotionField(PointND[][] variants, int context){
		this.context = context;
		warp = new IdentityTimeWarper();
		timeSamplePoints = variants.length;
		for (int t = 0; t < timeSamplePoints; t++) {
			timePoints.add(new ArrayList<PointND>());
			for (int i = 0; i < variants[t].length; i++){
				timePoints.get(t).add(variants[t][i]);
			}
		}
	}

	private int [] findCloseIndices(PointND position, int t0, int context){
		int [] closeIndex = new int[context];
		double [] dist = new double [context];
		for(int j =0; j<context; j++){
			dist[j] = (1.0 / (double)(context-j)) * Double.MAX_VALUE;
		}
		double maxDistance = Double.MAX_VALUE;
		for (int k = 0; k < timePoints.get(t0).size(); k++){			
			PointND test = timePoints.get(t0).get(k);
			double distance = position.euclideanDistance(test);
			//System.out.println(k + " " + distance);
			if (distance < maxDistance) {
				int maxIndex = -1;
				double max = -Double.MAX_VALUE;
				double lastMax = -Double.MAX_VALUE;
				for (int j =0; j<context; j++){
					if (dist[j] > max){
						lastMax = max;
						max = dist[j];
						maxIndex = j;
					}
				}
				maxDistance = (lastMax > distance)? lastMax : distance;
				dist[maxIndex] = distance;
				// pointIndex
				closeIndex[maxIndex] = k;
			}
		}
		return closeIndex;
	}

	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		double t0 = (warp.warpTime(initialTime) * (double)(timeSamplePoints-1));
		double t1 = (warp.warpTime(time) * (double)(timeSamplePoints-1));
		PointND revan = null;
		
		if ((Math.floor(t0) == t0)&&(Math.floor(t1) == t1)) {
			//simple case both values are extact hits in the time frames:
			int [] closeIndex1 = findCloseIndices(initialPosition, (int)Math.floor(t0), context);
			revan = getInterpolatedPoint(initialPosition, (int)t0, (int)t1, closeIndex1);
		} else {
			if ((Math.floor(t0) == t0)){ // at least the initial time is a direct hit (e.g. 0)
				int [] closeIndex1 = findCloseIndices(initialPosition, (int)Math.floor(t0), context);
				revan = getInterpolatedPoint(initialPosition, (int)Math.floor(t0), t1, closeIndex1);
			} else { // Most difficult case: no exact hit for t0 and t1 
				// really inefficient
				int [] closeIndex1 = findCloseIndices(initialPosition, (int)Math.floor(t0), context);
				int [] closeIndex2 = findCloseIndices(initialPosition, (int)Math.ceil(t0), context);
				revan = getInterpolatedPoint(initialPosition, t0, t1, closeIndex1, closeIndex2);
			}
		}
		return revan;
	}

	/**
	 * Method to interpolate a point if the initial Position is a direct hit on the interpolation grid of time t0, i.e. t0 is of type int.
	 * @param initialPosition the initalPosition
	 * @param t0 the initial time
	 * @param t1 the target time as double. will be interpolated by linear interpolation
	 * @param closeIndex the indices of the closest points in the interpolation grid at time t0
	 * @return the linear interpolated point
	 */
	private PointND getInterpolatedPoint(PointND initialPosition, int t0, double t1, int [] closeIndex){
		PointND one = getInterpolatedPoint(initialPosition, (int)t0, (int)Math.floor(t1), closeIndex);
		PointND two = getInterpolatedPoint(initialPosition, (int)t0, (int)Math.ceil(t1), closeIndex);
		one.getAbstractVector().multiplyBy(1.0-(t1-Math.floor(t1)));
		two.getAbstractVector().multiplyBy(1.0-(Math.ceil(t1) - t1));
		one.getAbstractVector().add(two.getAbstractVector());
		return one;
	}

	/**
	 * Method to interpolate a point if the initial position is not a direct hit on the interpolation grid, i.e. it is of type double.
	 * The target point will be determined by bilinear interpolation.
	 * @param initialPosition the initial position
	 * @param t0 the initial time
	 * @param t1 the target time
	 * @param closeIndex1 the close neighbors at time Math.floor(t0)
	 * @param closeIndex2 the close neighbors at time Math.ceil(t0)
	 * @return the bilinear interpolated point
	 */
	private PointND getInterpolatedPoint(PointND initialPosition, double t0, double t1, int [] closeIndex1, int []closeIndex2){
		PointND one = getInterpolatedPoint(initialPosition, (int)Math.floor(t0), (int)Math.floor(t1), closeIndex1);
		PointND two = getInterpolatedPoint(initialPosition, (int)Math.floor(t0), (int)Math.ceil(t1), closeIndex1);
		PointND three = getInterpolatedPoint(initialPosition, (int)Math.ceil(t0), (int)Math.floor(t1), closeIndex2);
		PointND four = getInterpolatedPoint(initialPosition, (int)Math.ceil(t0), (int)Math.ceil(t1), closeIndex2);
		one.getAbstractVector().multiplyBy(1.0-(t0 - Math.floor(t0)));
		two.getAbstractVector().multiplyBy(1.0-(t0 - Math.floor(t0)));
		three.getAbstractVector().multiplyBy(1.0-(Math.ceil(t0)-t0));
		four.getAbstractVector().multiplyBy(1.0-(Math.ceil(t0)-t0));
		one.getAbstractVector().multiplyBy(1.0-(t1-Math.floor(t1)));
		two.getAbstractVector().multiplyBy(1.0-(Math.ceil(t1) - t1));
		three.getAbstractVector().multiplyBy(1.0-(t1-Math.floor(t1)));
		four.getAbstractVector().multiplyBy(1.0-(Math.ceil(t1) - t1));
		one.getAbstractVector().add(two.getAbstractVector());
		one.getAbstractVector().add(three.getAbstractVector());
		one.getAbstractVector().add(four.getAbstractVector());
		return one;
	}

	private PointND getInterpolatedPoint(PointND initialPosition, int t0, int t1, int [] closeIndex){
		PointND p = initialPosition.clone();
		PointND [] close = new PointND[context];
		PointND [] dest = new PointND[context];
		double [] dist = new double [context];
		for (int i=0; i < context; i++){
			close[i] = timePoints.get(t0).get(closeIndex[i]);
			dest[i] = timePoints.get(t1).get(closeIndex[i]);
			dist[i] = p.euclideanDistance(close[i]);
		}
		SimpleVector shift = new SimpleVector(3);
		double sum = 0;
		for (int j =0; j < context; j++){
			double weight = 1000000;
			if (dist[j] != 0){
				weight = 1.0 / dist[j];
			}
			sum += weight;
			SimpleVector partialStep = SimpleOperators.subtract(dest[j].getAbstractVector(), close[j].getAbstractVector()).multipliedBy(weight);
			shift.add(partialStep);
		}
		p.getAbstractVector().add(shift.multipliedBy(1.0/sum));
		return p;
	}
	
	public ArrayList<PointND> getPositions(double initialTime,
			double time, PointND ... initialPositions) {
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i=0; i< initialPositions.length; i++){
			list.add(getPosition(initialPositions[i], initialTime, time));
		}
		return list;
	}

	public ArrayList<PointND> getPositions(PointND initialPoint, double initialTime, double ... times){
		ArrayList<PointND> points = new ArrayList<PointND>();
		double t0 = (warp.warpTime(initialTime) * (double)(timeSamplePoints-1));
		if ((Math.floor(t0) == t0)) {
			int [] closeIndex = findCloseIndices(initialPoint, (int)Math.floor(t0), context);
			for (double t:times) {
				double t1 = (warp.warpTime(t) * (double)(timeSamplePoints-1));
				if (Math.floor(t1) == t1) {
					points.add(getInterpolatedPoint(initialPoint, (int)Math.floor(t0), (int)Math.floor(t1), closeIndex));
				} else {
					points.add(getInterpolatedPoint(initialPoint, (int)Math.floor(t0), t1, closeIndex));
				}
			}
		} else {
			int [] closeIndex1 = findCloseIndices(initialPoint, (int)Math.floor(t0), context);
			int [] closeIndex2 = findCloseIndices(initialPoint, (int)Math.ceil(t0), context);
			for (double t:times) {
				double t1 = (warp.warpTime(t) * (double)(timeSamplePoints-1));
				points.add(getInterpolatedPoint(initialPoint, t0, t1, closeIndex1, closeIndex2));
			}
		}
		return points;
	}

	public TimeWarper getTimeWarper() {
		return warp;
	}

	public void setTimeWarper(TimeWarper warp) {
		this.warp = warp;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
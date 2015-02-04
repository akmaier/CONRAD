package edu.stanford.rsl.conrad.geometry.motion;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Models a MotionField as two points and a normal vector. The 
 * @author akmaier
 *
 */
public class CompressionMotionField extends SimpleMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = 191865158667227329L;
	private SimpleVector direction;
	private PointND min;
	private PointND max;
	private double length;
	
	public CompressionMotionField (PointND min, PointND max, SimpleVector direction){
		length = direction.normL2();
		this.direction = direction.dividedBy(length);
		this.min = min;
		this.max = max;
	}
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		PointND p = initialPosition.clone();
		if (
				p.get(0) >= min.get(0) &&
				p.get(1) >= min.get(1) &&
				p.get(2) >= min.get(2) &&
				p.get(0) <= max.get(0) &&
				p.get(1) <= max.get(1) &&
				p.get(2) <= max.get(2)){
			Plane3D movingBound = new Plane3D(min, direction.clone());
			double distance = movingBound.computeDistance(max);
			if (distance < 0){
				movingBound = new Plane3D(max, direction.clone());
				distance = movingBound.computeDistance(min);
			}
			double movingDistance = movingBound.computeDistance(p);
			double warpedInit = warp.warpTime(initialTime); 
			double warpedTime = warp.warpTime(time);
			double difference = warpedTime - warpedInit;
			p.getAbstractVector().add(direction.multipliedBy(((distance - movingDistance)/distance)*length*(difference)));
		}
		return p;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.motion;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to project a MotionField onto a plane. Any off-plane motion is inhibited.
 * 
 * @author akmaier
 *
 */
public class PlanarMotionField extends SimpleMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7264620135393212142L;
	private SimpleVector normal;
	private MotionField fullMotion;
	
	public PlanarMotionField (MotionField fullMotion, SimpleVector planeNormal){
		this.fullMotion = fullMotion;
		normal = planeNormal.normalizedL2();
//		normal = planeNormal;
//		normal.multiplyBy(1.0 / planeNormal.normL2());
	}
	
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		PointND newPosition = fullMotion.getPosition(initialPosition, initialTime, time);
		return convertToPlanarMotionPosition(initialPosition, newPosition);
	}
	
	private PointND convertToPlanarMotionPosition(PointND initialPosition, PointND newPosition){
		SimpleVector shift = SimpleOperators.subtract(newPosition.getAbstractVector(), initialPosition.getAbstractVector());
		SimpleVector inPlaneShift = General.crossProduct(normal, General.crossProduct(shift, normal));
		PointND newPositionPlanarMotion = initialPosition.clone();
		newPositionPlanarMotion.getAbstractVector().add(inPlaneShift);
		return newPositionPlanarMotion;
	}
	
	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition, double initialTime,
			double ... times) {
		ArrayList<PointND> newPositions = fullMotion.getPositions(initialPosition, initialTime, times);
		ArrayList<PointND> planarPositions = new ArrayList<PointND>();
		for (int i = 0; i < newPositions.size(); i++){
			planarPositions.add(convertToPlanarMotionPosition(initialPosition, newPositions.get(i)));
		}
		return planarPositions;
	}

	@Override
	public TimeWarper getTimeWarper(){
		return fullMotion.getTimeWarper();
	}
	
	@Override
	public void setTimeWarper(TimeWarper warp){
		fullMotion.setTimeWarper(warp);
	}

}

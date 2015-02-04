package edu.stanford.rsl.conrad.geometry.motion;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class MovingCenterRotationMotionField extends RotationMotionField {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4231093664698586329L;
	protected MotionField centerTransform;
	
	public MovingCenterRotationMotionField (PointND transformationCenter, MotionField centerTransform, SimpleVector rotationAxis, double angle){
		super(transformationCenter, rotationAxis, angle);
		this.centerTransform = centerTransform; 
	}	
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		back = new Translation(centerTransform.getPosition(center, initialTime, time).getAbstractVector());
		SimpleVector translateToInitialTime = centerTransform.getPosition(center, 0, initialTime).getAbstractVector();
		translateToInitialTime.negate();
		toCenter = new Translation(translateToInitialTime);
		Transform scaleRotation = getTransform(initialTime, time);
		PointND p = back.transform(scaleRotation.transform(toCenter.transform(initialPosition)));
		return p;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
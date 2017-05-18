package edu.stanford.rsl.conrad.geometry.motion;

import java.util.TreeMap;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.xcat.ViconAffineTransform;
import edu.stanford.rsl.conrad.utils.Configuration;

/**********************************************
 * Class to create an artificial motion field *
 * without VICON markers.                     *
 * @author Jang CHOI / Oleksiy Rybakov        *
 **********************************************/

public class ArtificialMotionField extends SimpleMotionField {
	
	private static final long serialVersionUID = -719945083192209758L;
	private PointND[] beforePts = new PointND[2];	
	private PointND[] afterPts = new PointND[2];
	private ViconAffineTransform tmpTransformation1 = null;
	private ViconAffineTransform tmpTransformation2 = null;
	private String part;
	private String splineTitle;
	private static TreeMap<String, TreeMap<Integer, Double>> boneScalingMemorizer;
	private static TreeMap<String, TreeMap<Integer, SimpleMatrix>> boneGlobalTransformMemorizer;
	
	// input parameters
	private double lengthLowerLeft;		// length of left tibia in mm
	private double lengthLowerRight;	// length of right tibia in mm
	private double lengthUpperLeft;		// length of left femur in mm
	private double lengthUpperRight;	// length of right femur in mm
	private double numberOfSquats;			// number of squats within measured time
	private double angleMin;			// minimal squatting angle between tibia and femur in degrees
	private double angleMax;			// maximal squatting angle tibia and femur in degrees
	
	// constructor
	public ArtificialMotionField(String transformPart, String spTitle, double lengthLowerLeft,
			double lengthLowerRight, double lengthUpperLeft, double lengthUpperRight, double numberOfSquats,
			double angleMin, double angleMax){
		this.lengthLowerLeft = lengthLowerLeft;
		this.lengthLowerRight = lengthLowerRight;
		this.lengthUpperLeft = lengthUpperLeft;
		this.lengthUpperRight = lengthUpperRight;
		this.numberOfSquats = numberOfSquats;
		
		// This is the half of the angle between tibia and femur,
		// not the medical knee flexion angle. The order between
		// angleMin and angleMax is now swapped.
		this.angleMin = 90 - angleMax * 0.5;
		this.angleMax = 90 - angleMin * 0.5;
		
		warp = new IdentityTimeWarper();
		part = transformPart;
		splineTitle = spTitle;
		if(boneScalingMemorizer == null){
			boneScalingMemorizer = new TreeMap<String, TreeMap<Integer,Double>>();
		}
		if(boneGlobalTransformMemorizer == null){
			boneGlobalTransformMemorizer = new TreeMap<String, TreeMap<Integer,SimpleMatrix>>();
		}
	}
	
	private void addElementToScalingMemorizer(String part, int projectionNr, double scaling){
		if(!boneScalingMemorizer.containsKey(part)){
			boneScalingMemorizer.put(part, new TreeMap<Integer,Double>());
		}
		boneScalingMemorizer.get(part).put(projectionNr, scaling);
	}
	
	private void addElementToTransformMemorizer(String part, int projectionNr, SimpleMatrix tform){
		if(!boneGlobalTransformMemorizer.containsKey(part)){
			boneGlobalTransformMemorizer.put(part, new TreeMap<Integer,SimpleMatrix>());
		}
		boneGlobalTransformMemorizer.get(part).put(projectionNr, tform);
	}
	
	/******************************************************
	 * A function to compute angle as a function of time. *
	 * @param t given time                                *
	 * @return angle in radians at time t                 *
	 ******************************************************/
	private double computeAngle(double t){
		return Math.toRadians(angleMin + 2 * (angleMax - angleMin) * Math.abs(0.5 * (1 - t) + (int) (0.5 * t)));
	}
	
	/*****************************************************
	 * Function to get position of organ at a given time *
	 * depending on initial position and initial time.   *
	 *****************************************************/
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time){
		
		// knee gap elimination
		boolean fKneeGapElimination = true;
		double kneeGapEliWeightKnee = 0.74;		// knee weight
		double kneeGapEliWeightAnkle = 0.77;	// ankle weight
		
		// register keys
		double projections = Configuration.getGlobalConfiguration().getGeometry().getNumProjectionMatrices();
		if(numberOfSquats * 2 + 1 > projections){
			// not enough projections
			// TODO This code has to be written as an exception.
			// I have not modified it so far to preserve the code structure.
			System.err.println("Not enough projections. This program is exited now.");
			System.exit(0);
		}
		
		// projection number and time parameter for angle computation
		int projectionNumber = (int) Math.round((warp.warpTime(time) - warp.warpTime(initialTime)) * projections - 1);
		PointND p1 = null;
		PointND p2 = null;
		double leftAxialRotDeg = -85;
		double rightAxialRotDeg = -90;
		PointND leftPts = new PointND();
		PointND rightPts = new PointND();
		
		// knee angle computation
		double t = projectionNumber * 2.0 * numberOfSquats / (projections - 1); // time step t' in report
		double angle = computeAngle(t); // knee angle in radians
		double cosAngle = Math.cos(angle); // store cos and sin to prevent re-computation
		double sinAngle = Math.sin(angle);
		
		// patella - this code is not modified
		if(splineTitle.toLowerCase().contains("l_patella") || splineTitle.toLowerCase().contains("leftpatella")){
			double anglePatellaVsFemor = General.toRadians(0.9 * Math.toDegrees(angle) - 1.84588);
			PointND leftPatellaRotCenter = new PointND(264.8, 264.6, -590.92);
			RotationMotionField leftFemurMotion = new RotationMotionField(leftPatellaRotCenter, new SimpleVector(1, 0, 0), anglePatellaVsFemor);
			initialPosition = leftFemurMotion.getPosition(initialPosition, 0, 1);
		} else if(splineTitle.toLowerCase().contains("r_patella") || splineTitle.toLowerCase().contains("rightpatella")){
			double anglePatellaVsFemor = General.toRadians(0.9 * Math.toDegrees(angle) - 1.84588);
			PointND rightPatellaRotCenter = new PointND(21.1, 269.1, -590.92);
			RotationMotionField rightFemurMotion = new RotationMotionField(rightPatellaRotCenter, new SimpleVector(1, 0, 0), anglePatellaVsFemor);
			initialPosition = rightFemurMotion.getPosition(initialPosition, 0, 1);
		}
		
		// other parts - ankle positions are hard coded
		if(part.equals("LeftLower")){
			beforePts[0] = new PointND(316.71, 293.02, -1011.74);
			beforePts[1] = new PointND(274.87, 275.59, -619.16);
			afterPts[0] = new PointND(-196.09, 23.465, 74.553);
			afterPts[1] = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
			if(fKneeGapElimination){
				leftPts = afterPts[1];
				rightPts = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
				afterPts[1].set(0, leftPts.get(0) * kneeGapEliWeightKnee + rightPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(1, leftPts.get(1) * kneeGapEliWeightKnee + rightPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(2, leftPts.get(2) * kneeGapEliWeightKnee + rightPts.get(2) * (1 - kneeGapEliWeightKnee));
				leftPts = afterPts[0];
				rightPts = new PointND(-198.98, 305.45, 72.421);
				afterPts[0].set(0, leftPts.get(0) * kneeGapEliWeightAnkle + rightPts.get(0) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(1, leftPts.get(1) * kneeGapEliWeightAnkle + rightPts.get(1) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(2, leftPts.get(2) * kneeGapEliWeightAnkle + rightPts.get(2) * (1 - kneeGapEliWeightAnkle));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			addElementToScalingMemorizer("LeftLower", projectionNumber, tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("LeftLower", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if(part.equals("RightLower")){
			beforePts[0] = new PointND(-28.12, 295.11, -1013.33);
			beforePts[1] = new PointND(10.90, 275.60, -620.46);
			afterPts[0] = new PointND(-198.98, 305.45, 72.421);
			afterPts[1] = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
			if(fKneeGapElimination){
				leftPts = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
				rightPts = afterPts[1];
				afterPts[1].set(0, rightPts.get(0) * kneeGapEliWeightKnee + leftPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(1, rightPts.get(1) * kneeGapEliWeightKnee + leftPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(2, rightPts.get(2) * kneeGapEliWeightKnee + leftPts.get(2) * (1 - kneeGapEliWeightKnee));
				leftPts = new PointND(-196.09, 23.465, 74.553);
				rightPts = afterPts[0];
				afterPts[0].set(0, rightPts.get(0) * kneeGapEliWeightAnkle + leftPts.get(0) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(1, rightPts.get(1) * kneeGapEliWeightAnkle + leftPts.get(1) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(2, rightPts.get(2) * kneeGapEliWeightAnkle + leftPts.get(2) * (1 - kneeGapEliWeightAnkle));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			addElementToScalingMemorizer("RightLower",projectionNumber,tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("RightLower", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if(part.equals("LeftUpper")){
			beforePts[0] = new PointND(264.52, 271.70, -632.25);
			beforePts[1] = new PointND(233.3, 259.99, -166.63);		
			afterPts[0] = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
			afterPts[1] = new PointND(-196.09 - lengthLowerLeft * cosAngle + lengthUpperLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle + lengthUpperLeft * sinAngle);
			if(fKneeGapElimination){
				leftPts = afterPts[0];
				rightPts = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
				afterPts[0].set(0, leftPts.get(0) * kneeGapEliWeightKnee + rightPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(1, leftPts.get(1) * kneeGapEliWeightKnee + rightPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(2, leftPts.get(2) * kneeGapEliWeightKnee + rightPts.get(2) * (1 - kneeGapEliWeightKnee));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			addElementToScalingMemorizer("LeftUpper", projectionNumber, tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("LeftUpper", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if(part.equals("RightUpper")){
			beforePts[0] = new PointND(22.60, 271.70, -632.9);
			beforePts[1] = new PointND(57.72, 259.99, -166.63);
			afterPts[0] = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
			afterPts[1] = new PointND(-198.98 - lengthLowerRight * cosAngle + lengthUpperRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle + lengthUpperRight * sinAngle);
			if(fKneeGapElimination){
				leftPts = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
				rightPts = afterPts[0];
				afterPts[0].set(0, rightPts.get(0) * kneeGapEliWeightKnee + leftPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(1, rightPts.get(1) * kneeGapEliWeightKnee + leftPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(2, rightPts.get(2) * kneeGapEliWeightKnee + leftPts.get(2) * (1 - kneeGapEliWeightKnee));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			addElementToScalingMemorizer("RightUpper",projectionNumber,tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("RightUpper", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if(part.equals("LeftDual")){
			double skinningWeightLimit = 70;
			double jointPositionInZaxis = -642;
			double ZLimitTop = jointPositionInZaxis + skinningWeightLimit;
			double ZLimitBottom = jointPositionInZaxis - skinningWeightLimit;
			double Zcurrent = initialPosition.get(2);
			beforePts[0] = new PointND(264.52, 271.70, -632.25);
			beforePts[1] = new PointND(233.3, 259.99, -166.63);
			afterPts[0] = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
			afterPts[1] = new PointND(-196.09 - lengthLowerLeft * cosAngle + lengthUpperLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle + lengthUpperLeft * sinAngle);
			if(fKneeGapElimination){
				leftPts = afterPts[0];
				rightPts = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
				afterPts[0].set(0, leftPts.get(0) * kneeGapEliWeightKnee + rightPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(1, leftPts.get(1) * kneeGapEliWeightKnee + rightPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(2, leftPts.get(2) * kneeGapEliWeightKnee + rightPts.get(2) * (1 - kneeGapEliWeightKnee));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			beforePts[0] = new PointND(316.71, 293.02, -1011.74);
			beforePts[1] = new PointND(274.87, 275.59, -619.16);
			afterPts[0] = new PointND(-196.09, 23.465, 74.553);
			afterPts[1] = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
			if(fKneeGapElimination){
				leftPts = afterPts[1];
				rightPts = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
				afterPts[1].set(0, leftPts.get(0) * kneeGapEliWeightKnee + rightPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(1, leftPts.get(1) * kneeGapEliWeightKnee + rightPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(2, leftPts.get(2) * kneeGapEliWeightKnee + rightPts.get(2) * (1 - kneeGapEliWeightKnee));
				leftPts = afterPts[0];
				rightPts = new PointND(-198.98, 305.45, 72.421);
				afterPts[0].set(0, leftPts.get(0) * kneeGapEliWeightAnkle + rightPts.get(0) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(1, leftPts.get(1) * kneeGapEliWeightAnkle + rightPts.get(1) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(2, leftPts.get(2) * kneeGapEliWeightAnkle + rightPts.get(2) * (1 - kneeGapEliWeightAnkle));
			}
			tmpTransformation2 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
			p2 = tmpTransformation2.getTransformedPoints(initialPosition);
			double weight = 0;
			if(Zcurrent >= ZLimitTop){
				weight = 1;
			} else if(Zcurrent >= ZLimitBottom){
				weight = (Zcurrent - ZLimitBottom) / (ZLimitTop - ZLimitBottom);
			}
			p1.set(0, p1.get(0) * weight + p2.get(0) * (1 - weight));
			p1.set(1, p1.get(1) * weight + p2.get(1) * (1 - weight));
			p1.set(2, p1.get(2) * weight + p2.get(2) * (1 - weight));
		} else if(part.equals("RightDual")){
			double skinningWeightLimit = 70;
			double jointPositionInZaxis = -642;
			double ZLimitTop = jointPositionInZaxis + skinningWeightLimit;
			double ZLimitBottom = jointPositionInZaxis - skinningWeightLimit;
			double Zcurrent = initialPosition.get(2);
			beforePts[0] = new PointND(22.60, 271.70, -632.9);
			beforePts[1] = new PointND(57.72, 259.99, -166.63);
			afterPts[0] = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
			afterPts[1] = new PointND(-198.98 - lengthLowerRight * cosAngle + lengthUpperRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle + lengthUpperRight * sinAngle);
			if(fKneeGapElimination){
				leftPts = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
				rightPts = afterPts[0];
				afterPts[0].set(0, rightPts.get(0) * kneeGapEliWeightKnee + leftPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(1, rightPts.get(1) * kneeGapEliWeightKnee + leftPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[0].set(2, rightPts.get(2) * kneeGapEliWeightKnee + leftPts.get(2) * (1 - kneeGapEliWeightKnee));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			beforePts[0] = new PointND(-28.12, 295.11, -1013.33);
			beforePts[1] = new PointND(10.90, 275.60, -620.46);
			afterPts[0] = new PointND(-198.98, 305.45, 72.421);
			afterPts[1] = new PointND(-198.98 - lengthLowerRight * cosAngle, 305.45, 72.421 + lengthLowerRight * sinAngle);
			if(fKneeGapElimination){
				leftPts = new PointND(-196.09 - lengthLowerLeft * cosAngle, 23.465, 74.553 + lengthLowerLeft * sinAngle);
				rightPts = afterPts[1];
				afterPts[1].set(0, rightPts.get(0) * kneeGapEliWeightKnee + leftPts.get(0) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(1, rightPts.get(1) * kneeGapEliWeightKnee + leftPts.get(1) * (1 - kneeGapEliWeightKnee));
				afterPts[1].set(2, rightPts.get(2) * kneeGapEliWeightKnee + leftPts.get(2) * (1 - kneeGapEliWeightKnee));
				leftPts = new PointND(-196.09, 23.465, 74.553);
				rightPts = afterPts[0];
				afterPts[0].set(0, rightPts.get(0) * kneeGapEliWeightAnkle + leftPts.get(0) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(1, rightPts.get(1) * kneeGapEliWeightAnkle + leftPts.get(1) * (1 - kneeGapEliWeightAnkle));
				afterPts[0].set(2, rightPts.get(2) * kneeGapEliWeightAnkle + leftPts.get(2) * (1 - kneeGapEliWeightAnkle));
			}
			tmpTransformation2 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
			p2 = tmpTransformation2.getTransformedPoints(initialPosition);
			double weight = 0;
			if(Zcurrent >= ZLimitTop){
				weight = 1;
			} else if(Zcurrent >= ZLimitBottom){
				weight = (Zcurrent - ZLimitBottom) / (ZLimitTop - ZLimitBottom);
			}
			p1.set(0, p1.get(0) * weight + p2.get(0) * (1 - weight));
			p1.set(1, p1.get(1) * weight + p2.get(1) * (1 - weight));
			p1.set(2, p1.get(2) * weight + p2.get(2) * (1 - weight));
		}
		return p1;
	}
	
	public TreeMap<String, TreeMap<Integer, Double>> getBoneScalingMemorizer() {
		return boneScalingMemorizer;
	}
	
	public TreeMap<String, TreeMap<Integer, SimpleMatrix>> getBoneGlobalTransformMemorizer() {
		return boneGlobalTransformMemorizer;
	}
}


/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

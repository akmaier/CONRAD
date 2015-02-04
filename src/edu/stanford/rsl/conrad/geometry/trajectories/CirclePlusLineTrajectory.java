/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.trajectories;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This trajectory can be used to create a circle plus line trajectory.
 * 
 * @author akmaier
 *
 */
public class CirclePlusLineTrajectory extends CircularTrajectory {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4500784187877049083L;
	
	public CirclePlusLineTrajectory() {
		super();
	}
	
	public CirclePlusLineTrajectory(Trajectory geometry) {
		super(geometry);
	}

	public void setTrajectory(int numProjectionMatricesCircle, double sourceToAxisDistance, double averageAngularIncrement, 
			double detectorOffsetX, double detectorOffsetY, CameraAxisDirection uDirection, 
			CameraAxisDirection vDirection, SimpleVector rotationAxis, PointND rotationCenter, double angleFirstProjection, double lineIncrement, int lineSteps) {
		//TODO: Add a parameter for the angle of the first projection. (E.g., this might be -100� instead of the currently defaulted 0�.)
		int numProjectionMatrices = numProjectionMatricesCircle + lineSteps; 
		this.projectionMatrices = new Projection[numProjectionMatrices];
		this.primaryAngles = new double[numProjectionMatrices];
		this.numProjectionMatrices = numProjectionMatrices;
		this.sourceToAxisDistance = sourceToAxisDistance;
		this.averageAngularIncrement = averageAngularIncrement;

		double cosPhi = Math.cos(General.toRadians(angleFirstProjection));
		double sinPhi = Math.sin(General.toRadians(angleFirstProjection));
		SimpleMatrix rotMat = new SimpleMatrix(3,3);
		rotMat.setElementValue(0,0, cosPhi);
		rotMat.setElementValue(0, 1, sinPhi);
		rotMat.setElementValue(1,0,-sinPhi);
		rotMat.setElementValue(1, 1, cosPhi);
		rotMat.setElementValue(2, 2, 1);
		SimpleVector centerToCameraIdealAtInitialAngle = SimpleOperators.multiply(rotMat, new SimpleVector(sourceToAxisDistance, 0, 0));
		Plane3D trajPlane = new Plane3D(rotationAxis,SimpleOperators.multiplyInnerProd(rotationAxis, rotationCenter.getAbstractVector()));
		double distToPlane = trajPlane.computeDistance(new PointND(centerToCameraIdealAtInitialAngle));
		SimpleVector centerToCameraDir = SimpleOperators.subtract(SimpleOperators.add(rotationAxis.multipliedBy(-1*distToPlane),centerToCameraIdealAtInitialAngle),rotationCenter.getAbstractVector());
		centerToCameraDir.divideBy(centerToCameraDir.normL2());
		SimpleVector centerToCameraInitialInPlane = centerToCameraDir.multipliedBy(sourceToAxisDistance);

		for (int i=0; i< numProjectionMatricesCircle; i++){
			primaryAngles[i] = i*averageAngularIncrement + angleFirstProjection;
			//System.out.println(primaryAngles[i] + " " + averageAngularIncrement + " " + this.reconDimensions[0] + " " + this.reconDimensions[1]);
			projectionMatrices[i]= new Projection();
			double rotationAngle = General.toRadians(primaryAngles[i]);
			projectionMatrices[i].setRtFromCircularTrajectory(rotationCenter.getAbstractVector(), rotationAxis, sourceToAxisDistance, centerToCameraInitialInPlane, uDirection, vDirection, rotationAngle);
			SimpleVector spacingUV = new SimpleVector(pixelDimensionX, pixelDimensionY);
			SimpleVector sizeUV = new SimpleVector(detectorWidth, detectorHeight);
			SimpleVector offset = new SimpleVector(detectorOffsetX, detectorOffsetY);	
			projectionMatrices[i].setKFromDistancesSpacingsSizeOffset(sourceToDetectorDistance, spacingUV, sizeUV, offset, 1.0, 0);

		}
		for (int i=numProjectionMatricesCircle; i< numProjectionMatrices; i++){
			primaryAngles[i] = numProjectionMatricesCircle*averageAngularIncrement + angleFirstProjection;
			//System.out.println(primaryAngles[i] + " " + averageAngularIncrement + " " + this.reconDimensions[0] + " " + this.reconDimensions[1]);
			projectionMatrices[i]= new Projection();
			double rotationAngle = General.toRadians(primaryAngles[i]);
			projectionMatrices[i].setRtFromCircularTrajectory(rotationCenter.getAbstractVector(), rotationAxis, sourceToAxisDistance, centerToCameraInitialInPlane, uDirection, vDirection, rotationAngle);
			SimpleVector translation = projectionMatrices[i].getT();
			SimpleVector rotatedAxis = SimpleOperators.multiply(projectionMatrices[i].getR(), rotationAxis);
			translation.add(rotatedAxis.multipliedBy((i-numProjectionMatricesCircle+1)*lineIncrement));
			projectionMatrices[i].setTVector(translation);
			SimpleVector spacingUV = new SimpleVector(pixelDimensionX, pixelDimensionY);
			SimpleVector sizeUV = new SimpleVector(detectorWidth, detectorHeight);
			SimpleVector offset = new SimpleVector(detectorOffsetX, detectorOffsetY);	
			projectionMatrices[i].setKFromDistancesSpacingsSizeOffset(sourceToDetectorDistance, spacingUV, sizeUV, offset, 1.0, 0);

		}
		this.projectionStackSize = numProjectionMatrices;
		
		//System.out.println("Defined geometry with SDD " +sourceToDetectorDistance);
	}

}

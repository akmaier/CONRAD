/*
 * Copyright (C) 2010-2014 Andreas Maier
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
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

public class CircularTrajectory extends Trajectory {

	private static final long serialVersionUID = -3236098993706829039L;

	public CircularTrajectory() {
		super();
	}

	public CircularTrajectory(Trajectory source) {
		super(source);
		//		this.numProjectionMatrices = source.numProjectionMatrices;
		//		this.detectorWidth = source.detectorWidth;
		//		this.detectorHeight = source.detectorHeight;
		//		this.pixelDimensionX = source.pixelDimensionX;
		//		this.pixelDimensionY = source.pixelDimensionY;
		//		this.sourceToDetectorDistance = source.sourceToDetectorDistance;
		//		this.sourceToAxisDistance = source.sourceToAxisDistance;
		//		this.averageAngularIncrement = source.averageAngularIncrement;
		//		this.reconDimensions = Arrays.copyOf(source.reconDimensions, source.reconDimensions.length);
		//		this.reconVoxelSizes = Arrays.copyOf(source.reconVoxelSizes, source.reconVoxelSizes.length);
		//		this.projectionMatrices = source.projectionMatrices;
		//		this.primaryAngles = source.primaryAngles;
		//		this.projectionStackSize = source.projectionStackSize;
		//		this.originInPixelsX = source.originInPixelsX;
		//		this.originInPixelsY = source.originInPixelsY;
		//		this.originInPixelsZ = source.originInPixelsZ;
		//		this.detectorUDirection = source.detectorUDirection;
		//		this.detectorVDirection = source.detectorVDirection;
		//		this.rotationAxis = source.rotationAxis;
	}

	public void setTrajectory(int numProjectionMatrices, double sourceToAxisDistance, double averageAngularIncrement, 
			double detectorOffsetX, double detectorOffsetY, CameraAxisDirection uDirection, 
			CameraAxisDirection vDirection, SimpleVector rotationAxis) {
		this.setTrajectory(numProjectionMatrices, sourceToAxisDistance, averageAngularIncrement, 
				detectorOffsetX, detectorOffsetY, uDirection, vDirection, rotationAxis, 
				new PointND(0,0,0), 0);
	}
	
	public void setTrajectory(int numProjectionMatrices, double sourceToAxisDistance, double averageAngularIncrement, 
			double detectorOffsetX, double detectorOffsetY, CameraAxisDirection uDirection, 
			CameraAxisDirection vDirection, SimpleVector rotationAxis, PointND rotationCenter, double angleFirstProjection) {
		this.projectionMatrices = new Projection[numProjectionMatrices];
		this.primaryAngles = new double[numProjectionMatrices];
		this.numProjectionMatrices = numProjectionMatrices;
		this.sourceToAxisDistance = sourceToAxisDistance;
		this.averageAngularIncrement = averageAngularIncrement;
		this.detectorOffsetU = detectorOffsetX;
		this.detectorOffsetV = detectorOffsetY;

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

		for (int i=0; i< numProjectionMatrices; i++){
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
		this.projectionStackSize = numProjectionMatrices;
		//System.out.println("Defined geometry with SDD " +sourceToDetectorDistance);
	}
	

	public static void main(String[] args) {
		CONRAD.setup();
		Configuration config = Configuration.getGlobalConfiguration();
		CircularTrajectory traj = new CircularTrajectory(config.getGeometry());
		double[] startAngles = new double[]{0,20,40};
		for (int j = 0; j < startAngles.length; j++) {
			traj.setTrajectory(2,600,90,0,0,CameraAxisDirection.ROTATIONAXIS_PLUS,CameraAxisDirection.DETECTORMOTION_PLUS,
					new SimpleVector(0,0,1),new PointND(0,0,0), startAngles[j]);
			for (int i = 0; i < traj.getNumProjectionMatrices(); i++) {
				System.out.println("Matrix: " + traj.getProjectionMatrix(i).toString());
			}
			System.out.println(" ");
		}
	}

}

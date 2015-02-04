package edu.stanford.rsl.conrad.geometry.trajectories;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;

public class ExtrapolatedTrajectory extends
		Trajectory {

	/**
	 * 
	 */
	private static final long serialVersionUID = 800597336777372574L;
	private boolean debug = false;

	
	/**
	 * @return the debug
	 */
	public boolean isDebug() {
		return debug;
	}

	/**
	 * @param debug the debug to set
	 */
	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	public ExtrapolatedTrajectory(){
		super();
	}
	
	public ExtrapolatedTrajectory (Trajectory source){
		super(source);
	}
	
	/**
	 * Computes a rotation matrix around z axis in homogeneous coordinates. Angle is supposed to be in radians.
	 * @param angularIncrement
	 * @return the rotation matrix
	 */
	public static SimpleMatrix getHomogeneousRotationMatrixZ(double angularIncrement){
		double [][] transitionLeft = new double [4][4];
		double cos = Math.cos(angularIncrement);
		double sin = Math.sin(angularIncrement);
		// Rotation around z axis with increment of averageAngularIncrement in homogeneous coordinates:
		transitionLeft[0][0]= cos;
		transitionLeft[0][1]= -sin;
		transitionLeft[1][0]= sin;
		transitionLeft[1][1]= cos;
		transitionLeft[2][2]= 1;
		transitionLeft[3][3] =1;
		return new SimpleMatrix(transitionLeft);
	}
	
	public void extrapolateProjectionGeometry(){
		double fanAngle = Math.atan(((this.detectorWidth * pixelDimensionX)) / sourceToDetectorDistance)  / Math.PI * 180;
		double [] minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
		double range = minmax[1] - minmax[0];
		double minimumRange = 180 + fanAngle;
		double notCovered = minimumRange - range;
		if (notCovered > 0) { // too bad. We have less projections than the minimal set. So we have to fix this!
			// compute number of missing projections in each direction;
			int numSteps = (int) (Math.ceil((notCovered / averageAngularIncrement) / 2))+1;
			int newProjectionNumber = numProjectionMatrices + (numSteps *  2);
			Projection [] newMatrices = new Projection[newProjectionNumber];
			double [] newAngles = new double[newProjectionNumber];
			double radIncrement = averageAngularIncrement / 180 * Math.PI;
			SimpleMatrix transitionLeft = ExtrapolatedTrajectory.getHomogeneousRotationMatrixZ(radIncrement);
			SimpleMatrix transitionRight = ExtrapolatedTrajectory.getHomogeneousRotationMatrixZ(-radIncrement);
			// set references to the known area to the existing matrixes
			for (int i = 0; i < numProjectionMatrices; i++){
				newMatrices[i + numSteps] = projectionMatrices[i];
				newAngles[i + numSteps] = primaryAngles[i];
			}
			// Move the matrices towards the unknown area on the left side
			for (int i = 0; i < numSteps; i++){
				newMatrices[numSteps - i -1] = new Projection(SimpleOperators.multiplyMatrixProd(newMatrices[numSteps - i].computeP(), transitionLeft));
				newAngles[numSteps - i -1] = newAngles[numSteps - i] - averageAngularIncrement;
			}
			// Move the matrices toward the unknown area on the right side
			for (int i = 0; i < numSteps; i++){
				newMatrices[numSteps + i + numProjectionMatrices] = new Projection(SimpleOperators.multiplyMatrixProd(newMatrices[numSteps + i + numProjectionMatrices -1].computeP(),transitionRight));
				newAngles[numSteps + i + numProjectionMatrices] = newAngles[numSteps + i + numProjectionMatrices -1] + averageAngularIncrement;
			}
			projectionMatrices = newMatrices;
			numProjectionMatrices = newProjectionNumber;
			primaryAngles = newAngles;
		}
	}



	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

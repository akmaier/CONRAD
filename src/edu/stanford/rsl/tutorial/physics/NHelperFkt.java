/*
 * Copyright (C) 2018 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public abstract class NHelperFkt {

	/**
	 * Helper Function calculating the cross product of 2 3D vectors a x b
	 * Also known as: vector product or directed area product
	 * 
	 * 
	 * Given two linearly independent vectors a and b, the cross product, a × b, is a vector that is perpendicular to both a and b and thus normal to the plane containing them
	 */
	public static SimpleVector crossProduct3D(SimpleVector vec_a, SimpleVector vec_b) {

		assert vec_a.getLen() == 3 : new IllegalArgumentException("Length has to be greater than or equal to zero!");
		assert vec_b.getLen() == 3 : new IllegalArgumentException("Length has to be greater than or equal to zero!");

		double ret0 = vec_a.getElement(1) * vec_b.getElement(2) - vec_a.getElement(2) * vec_b.getElement(1);
		double ret1 = vec_a.getElement(0) * vec_b.getElement(2) - vec_a.getElement(2) * vec_b.getElement(0);
		double ret2 = vec_a.getElement(0) * vec_b.getElement(1) - vec_a.getElement(1) * vec_b.getElement(0);

		return new SimpleVector(ret0, ret1, ret2);
	}
	
	/**
	 * A simple function calculating the angle between two vectors
	 * 
	 * @return Value of the angle in radiant (Bogenmaß)
	 */
	public static double getAngleInRad(SimpleVector vec_a, SimpleVector vec_b) {
		double innerProd = SimpleOperators.multiplyInnerProd(vec_a.normalizedL2(), vec_b.normalizedL2());
		return Math.acos(clamp(innerProd, -1.0, 1.0));
	}
	/**
	 * A simple function calculating the angle between two vectors
	 * 
	 * @return Value of the angle in degree (Gradmaß)
	 */
	public static double getAngleInDeg(SimpleVector vec_a, SimpleVector vec_b) {
		return Math.toDegrees(getAngleInRad(vec_a, vec_b));
	}
		
	/**
	 * Check if a point x lies between the points a and b, when it is already know
	 * that all three points are on a straight line.
	 * 
	 * @return If x is between a and b
	 */
	public static boolean isBetween(SimpleVector a, SimpleVector b, SimpleVector x) {
		SimpleVector bminusa = b.clone();
		bminusa.subtract(a);
		
		SimpleVector xminusa = x.clone();
		xminusa.subtract(a);

		double dot = SimpleOperators.multiplyInnerProd(bminusa, xminusa);
		if (dot < 0)
			return false;

		double squaredlength = bminusa.getElement(0) * bminusa.getElement(0)
				+ bminusa.getElement(1) * bminusa.getElement(1) + bminusa.getElement(2) * bminusa.getElement(2);
		return dot <= squaredlength;
	}
	
	/**
	 * Create a tangent frame with the n vector as its origin 
	 * @param n = origin
	 * @return vector orthogonal to the original
	 */
	public static SimpleVector tangentFrame(SimpleVector n) {
		double nx = n.getElement(0);
		double ny = n.getElement(1);
		double nz = n.getElement(2);
		
	    if (Math.abs(nx) > Math.abs(ny)) {
	    	double wurzelX = 1.0 / Math.sqrt(nx * nx + nz * nz);
	    	return new SimpleVector(-nz * wurzelX, 0.0, nx * wurzelX);
	    } else {
	    	double wurzelY = 1.0 / Math.sqrt(ny * ny + nz * nz);
	    	return new SimpleVector(0.0, nz * wurzelY, -ny * wurzelY);
	    }

	}
	
	/**
	 * System output: Probability to scatter in a distinct direction
	 * @param energyEV = energy of the incident photon
	 * @param stepSize = distinguish between angles 
	 */
	public static void printComptonAngles(double energyEV, double stepSize) {
		System.out.println("\nPrinting all probabilities for the energy level " +  energyEV + " with step size " + stepSize);
		for(double i = 0; i <= 180.0; i+=stepSize) {
			double prob = XRayTracerSampling.comptonAngleCrossSection(energyEV, Math.toRadians(i));
			System.out.println("The Probability to scatter in the angle(" + i + ") is "+ String.format("%2.2f", prob*100.0) + "%");
		};
	}
	
	/**
	 * TODO: Still seems to contain an error. DO NOT USE
	 */
	public static SimpleVector toGlobal(SimpleVector randomUnitVec, SimpleVector normal) {
		
		double vx = randomUnitVec.getElement(0);
		double vy = randomUnitVec.getElement(1);
		double vz = randomUnitVec.getElement(2);
		
		SimpleVector normaleWorld = (normal.clone());
		//normaleWorld.divideBy(normaleWorld.getLen());
		normaleWorld.normalizeL2();
		//System.out.println("RandomUnitVec: " +  randomUnitVec + "\n NormaleInWorld: " + normal + "\n normalizeNorm: " + normaleWorld + " " + normaleWorld.normL2()  + "\n");
		
		SimpleVector t = tangentFrame(normaleWorld);
		SimpleVector b = crossProduct3D(normaleWorld, t);

	    SimpleVector a = t.multipliedBy(vx);
	    SimpleVector o = b.multipliedBy(vy);
	    SimpleVector e = normaleWorld.multipliedBy(vz);
	    a.add(o);
	    a.add(e);
	    
	    return a;
	}
	
	/**
	 * Rotate the reference frame such that the original Z-axis will lie in the direction of ref
	 * @see http://proj-clhep.web.cern.ch/proj-clhep/manual/UserGuide/VectorDefs/node49.html
	 * @param randomUnitVec
	 * @param normaleWorld
	 * @return The rotated vector
	 */
	public static SimpleVector transformToWorldCoordinateSystem(SimpleVector randomUnitVec, SimpleVector normaleWorld) {
		normaleWorld.normalizeL2();
		 
		double ux = normaleWorld.getElement(0);
		double uy = normaleWorld.getElement(1);
		double uz = normaleWorld.getElement(2);
		
		double uPar = ux*ux + uy*uy;
		if (uPar > 0){
			uPar = Math.sqrt(uPar);
			SimpleMatrix mat = new SimpleMatrix(3,3);
			mat.setColValue(0, new SimpleVector(ux*uz/uPar, uy*uz/uPar, -uPar));
			mat.setColValue(1, new SimpleVector(-uy/uPar, ux/uPar, 0));
			mat.setColValue(2, normaleWorld);
			SimpleVector a = SimpleOperators.multiply(mat,randomUnitVec);
			return a;
		}
		else if (normaleWorld.getElement(2) < 0){
			//rotate by 180 degrees about the y axis
			return new SimpleVector(-randomUnitVec.getElement(0),randomUnitVec.getElement(1),-randomUnitVec.getElement(2));
		}
		return randomUnitVec;
	}
	
	/**
	 * If val compares less than min, returns min; otherwise if max compares less than val, returns max; otherwise returns val.
	 * @param val = the value to clamp 
	 * @param min, max = boundaries to clamp v to 
	 */
	public static double clamp(double val, double min, double max) {
		return Math.max(min, Math.min(max, val));
	}
	
	/**
	 * If val compares less than min, returns min; otherwise if max compares less than val, returns max; otherwise returns val.
	 * @param val = the value to clamp 
	 * @param min, max = boundaries to clamp v to 
	 */
	public static float clamp(float val, float min, float max) {
	    return Math.max(min, Math.min(max, val));
	}
	
	/**
	 * @param angleRad = angle
	 * @return angle difference between 180 degree
	 */
	public static double transformToScatterAngle(double angleRad) {
		if(angleRad > Math.PI) {
			System.err.println("An angle between two vectors should not be > 180 deg!");
		}		
		//Angle of 180 is a deflection of 0 -> theta must be between 0 and pi
		return Math.PI - angleRad;
		
	}
	
	/**
	 * Computes the average energy of each pixel in the grid
	 * @return returns a grid where each pixel has the value of the average pixel in the original
	 */
	public static Grid2D compareToAveragePixel(Grid2D grid) {
		int width  = grid.getWidth();
		int height = grid.getHeight();
	
		Grid2D newGrid = new Grid2D(width, height);
		
		double totalEnergy = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				totalEnergy += grid.getAtIndex(x, y);
			}
		}
		//Average per pixel
		totalEnergy = totalEnergy  / (width * height);
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float val = grid.getAtIndex(x, y) / (float) totalEnergy;
				newGrid.setAtIndex(x, y, val);
			}
		}
		return newGrid;
	}
	
	/**
	 * Computes the root mean square error of two given images
	 * @param first images to compare
	 * @param second images to compare
	 * @return root mean square error
	 */
	public static double computeRMSE(Grid2D first, Grid2D second) {
		int width = first.getSize()[0];
		int height = first.getSize()[1];

		assert (width == second.getSize()[0]);
		assert (height == second.getSize()[1]);

		double totalDiff = 0.d;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				double diff = first.getAtIndex(x, y) - second.getAtIndex(x, y);
				totalDiff += diff * diff;
			}
		}

		return Math.sqrt(totalDiff / (width * height));
	}
	
}

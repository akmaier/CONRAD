package edu.stanford.rsl.conrad.geometry;


import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;


public abstract class Rotations {

	public enum BasicAxis {
		X_AXIS,
		Y_AXIS,
		Z_AXIS
	}
	
	/**
	 * Computes the rotation matrix from a to b.
	 * Note that both vectors must have the same length.
	 * 
	 * @param a the Vector a
	 * @param b the Vector b
	 * @return the rotation matrix from a to b
	 */
	public static SimpleMatrix getRotationMatrixFromAtoB(SimpleVector a, SimpleVector b){
		SimpleVector normal = General.crossProduct(a, b);
		double lenA = a.normL2();
		double lenB = b.normL2();
		if (Math.abs(lenA - lenB) > CONRAD.FLOAT_EPSILON) {
			throw new RuntimeException("Vector must have same length!");
		}
		double angle = Math.asin(normal.normL2() / (lenA*lenB));
		SimpleMatrix rot = Rotations.createRotationMatrixAboutAxis(new Axis(normal), angle);
		SimpleVector afterRotation = SimpleOperators.multiply(rot, a);
		if (General.euclideanDistance(afterRotation, b) < CONRAD.FLOAT_EPSILON) {
			return rot;
		} else {
			return Rotations.createRotationMatrixAboutAxis(new Axis(normal), -angle);
		}
	}

	/**
	 * Computes the angle (in radians) of the rotation from a to b (in the plane that is defined by (0,0,0), a, b).
	 * Note that both vectors must have the same length.
	 * 
	 * @param a the Vector a
	 * @param b the Vector b
	 * @return the rotation matrix from a to b
	 */
	public static double getRotationFromAtoB(SimpleVector a, SimpleVector b){
		SimpleVector ab = a.clone();
		ab.add(b.negated());
		if (ab.normL2() < CONRAD.FLOAT_EPSILON){
			return 0;
		} else {
			if (Math.abs(ab.normL2()-2.0) < CONRAD.FLOAT_EPSILON){
				return Math.PI;
			}
		}
		SimpleVector normal = General.crossProduct(a, b);
		double lenA = a.normL2();
		double lenB = b.normL2();
		if (Math.abs(lenA - lenB) > CONRAD.FLOAT_EPSILON) {
			throw new RuntimeException("Vector must have same length: " + lenA + " " + lenB);
		}
		double angle = Math.asin(normal.normL2() / (lenA*lenB));
		SimpleMatrix rot = Rotations.createRotationMatrixAboutAxis(new Axis(normal), angle);
		SimpleVector afterRotation = SimpleOperators.multiply(rot, a);
		@SuppressWarnings("unused")
		double angle1 = General.toDegrees(angle);
		if (General.euclideanDistance(afterRotation, b) < CONRAD.FLOAT_EPSILON) {
			return angle;
		} else {
			rot = Rotations.createRotationMatrixAboutAxis(new Axis(normal), -angle);
			afterRotation = SimpleOperators.multiply(rot, a);
			@SuppressWarnings("unused")
			double angle2 = General.toDegrees(-angle);
			if (General.euclideanDistance(afterRotation, b) < CONRAD.FLOAT_EPSILON) {
				return -angle;
			} else {
				rot = Rotations.createRotationMatrixAboutAxis(new Axis(normal), Math.PI-angle);
				afterRotation = SimpleOperators.multiply(rot, a);
				if (General.euclideanDistance(afterRotation, b) < CONRAD.FLOAT_EPSILON) {
					return Math.PI-angle;
				} else {
					rot = Rotations.createRotationMatrixAboutAxis(new Axis(normal), Math.PI+angle);
					afterRotation = SimpleOperators.multiply(rot, a);
					System.out.println("Problem");
					return Double.MAX_VALUE;
				}
			}
			
		}
	}
	
	
	public static SimpleMatrix createBasicRotationMatrix(final BasicAxis axis, double angle) {
		final double s = Math.sin(angle); 
		final double c = Math.cos(angle); 
		if (axis == BasicAxis.X_AXIS) return new SimpleMatrix(new double[][] {
				{1.0, 0.0, 0.0},
				{0.0,  c , -s },
				{0.0,  s ,  c }
		});
		else if (axis == BasicAxis.Y_AXIS) return new SimpleMatrix(new double[][] {
				{ c , 0.0,  s },
				{0.0, 1.0, 0.0},
				{-s , 0.0,  c }
		});
		else if (axis == BasicAxis.Z_AXIS) return new SimpleMatrix(new double[][] {
				{ c , -s , 0.0},
				{ s ,  c , 0.0},
				{0.0, 0.0, 1.0}
		});
		else throw new RuntimeException("Unknown axis!");
	}
	
	public static SimpleMatrix createBasicRotationMatrixDerivative(final BasicAxis axis, double angle) {
		final double s = Math.sin(angle); 
		final double c = Math.cos(angle); 
		if (axis == BasicAxis.X_AXIS) return new SimpleMatrix(new double[][] {
				{0.0, 0.0, 0.0},
				{0.0,  -s , -c },
				{0.0,  c ,  -s }
		});
		else if (axis == BasicAxis.Y_AXIS) return new SimpleMatrix(new double[][] {
				{ -s , 0.0,  c },
				{0.0, 0.0, 0.0},
				{-c , 0.0,  -s }
		});
		else if (axis == BasicAxis.Z_AXIS) return new SimpleMatrix(new double[][] {
				{ -s , -c , 0.0},
				{ c ,  -s , 0.0},
				{0.0, 0.0, 0.0}
		});
		else throw new RuntimeException("Unknown axis!");
	}
	
	/**
	 * Creates a rotation matrix derivative w.r.t. the given basic axis and given angles
	 * 
	 * @param axis the axis to derive for
	 * @param angleX the angle in X
	 * @param angleY the angle in Y
	 * @param angleZ the angle in Z
	 * @return the matrix
	 */
	public static SimpleMatrix createRotationMatrixDerivative(BasicAxis axis, double angleX, double angleY, double angleZ){
		if (axis == BasicAxis.X_AXIS){
			SimpleMatrix xrot = createBasicRotationMatrixDerivative(axis, angleX);
			SimpleMatrix xyrot = SimpleOperators.multiplyMatrixProd(xrot, createBasicYRotationMatrix(angleY));
			return SimpleOperators.multiplyMatrixProd(xyrot, createBasicZRotationMatrix(angleZ));
		}
		else if (axis == BasicAxis.Y_AXIS){
			SimpleMatrix xrot = createBasicXRotationMatrix(angleX);
			SimpleMatrix xyrot = SimpleOperators.multiplyMatrixProd(xrot, createBasicRotationMatrixDerivative(axis, angleY));
			return SimpleOperators.multiplyMatrixProd(xyrot, createBasicZRotationMatrix(angleZ));
		}
		else if (axis == BasicAxis.Z_AXIS){
			SimpleMatrix xrot = createBasicXRotationMatrix(angleX);
			SimpleMatrix xyrot = SimpleOperators.multiplyMatrixProd(xrot, createBasicYRotationMatrix(angleY));
			return SimpleOperators.multiplyMatrixProd(xyrot, createBasicRotationMatrixDerivative(axis,angleZ));
		}
		else throw new RuntimeException("Unknown axis!");	
	}
	
	/**
	 * Creates a rotation matrix as the product of 
	 * RotationMatrixX * RotationMatrixY * RotationMatrixZ
	 * 
	 * @param angleX the angle in X
	 * @param angleY the angle in Y
	 * @param angleZ the angle in Z
	 * @return the matrix
	 */
	public static SimpleMatrix createRotationMatrix(double angleX, double angleY, double angleZ){
		SimpleMatrix xrot = createBasicXRotationMatrix(angleX);
		SimpleMatrix xyrot = SimpleOperators.multiplyMatrixProd(xrot, createBasicYRotationMatrix(angleY));
		return SimpleOperators.multiplyMatrixProd(xyrot, createBasicZRotationMatrix(angleZ));
	}
	
	
	public static SimpleMatrix createBasicXRotationMatrix(double angle){
		return createBasicRotationMatrix(BasicAxis.X_AXIS, angle);
	}
	public static SimpleMatrix createBasicYRotationMatrix(double angle){
		return createBasicRotationMatrix(BasicAxis.Y_AXIS, angle);
	}
	public static SimpleMatrix createBasicZRotationMatrix(double angle){
		return createBasicRotationMatrix(BasicAxis.Z_AXIS, angle);
	}

	
	/**
	 * 
	 * @param axis the direction of the axis (can be any length)
	 * @param angle the angle in radians.
	 * @return the Rotation Matrix
	 */
	public static SimpleMatrix createRotationMatrixAboutAxis(final SimpleVector axis, double angle){
		return createRotationMatrixAboutAxis(new Axis(axis), angle);
	}
	
	/**
	 * Creates a Rotation Matrix about an arbitrary axis.
	 * @param axis  Axis of Rotation
	 * @param angle in radians
	 * @return rotation matrix
	 */
	public static SimpleMatrix createRotationMatrixAboutAxis(Axis axis, double angle){
		final SimpleVector axisVec = axis.getAxisVector();
		assert (Math.abs(axisVec.normL2() - 1.0) < Math.sqrt(CONRAD.DOUBLE_EPSILON));
		final double x = axisVec.getElement(0), y = axisVec.getElement(1), z = axisVec.getElement(2);
		final double s = Math.sin(angle);
		final double c = Math.cos(angle);
		final double omc = 1 - c;
		
		return new SimpleMatrix(new double[][] {
				{x*x*omc + c,   x*y*omc - z*s, x*z*omc + y*s},
				{x*y*omc + z*s, y*y*omc + c,   y*z*omc - x*s},
				{x*z*omc - y*s, y*z*omc + x*s, z*z*omc + c}
		});
	}
	
	/**
	 * Calculates rotational change of axis matrix from old system to new system using directional cosines.
	 * @param oldSystem Old coordinate System
	 * @param newSystem New Coordinate System
	 * @return change of coordinate matrix
	 */
	public static SimpleMatrix create3DChangeOfAxesMatrix(CoordinateSystem oldSystem, CoordinateSystem newSystem){
		if(oldSystem.dimension() != newSystem.dimension()){
			throw new RuntimeException("Incompartible Coordinate Systems");
		}
		SimpleMatrix rotator = new SimpleMatrix(oldSystem.dimension(), oldSystem.dimension());
		
		Axis [] newAxes = newSystem.Axes();	
		Axis [] oldAxes = oldSystem.Axes();	
		
		for(int i = 0; i < newAxes.length; i++){
			SimpleVector currAxis = newAxes[i].getAxisVector();			
			for(int j = 0; j < newAxes.length; j++){
				rotator.addToElement(i, j, SimpleOperators.multiplyInnerProd(currAxis,oldAxes[j].getAxisVector()));
			}
		}
		
		return rotator.transposed();
	}
	



	


}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

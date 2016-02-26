/*
 * Copyright (C) 2015 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.basics;

import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.Rotations.BasicAxis;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;

/**
 * Simple Class to show rotation functionalities using matrices.
 * 
 * @author akmaier
 *
 */
public class RotationMatrixExamples {

	public static void main(String[] args) {
		// Create 90 degree rotation about Z
		SimpleMatrix rotZ = Rotations.createBasicZRotationMatrix(Math.PI/180*-142);
		// Create 90 degree rotation about Y
		SimpleMatrix rotY = Rotations.createBasicYRotationMatrix(Math.PI/180*90);
		// Create 90 degree rotation about X
		SimpleMatrix rotX = Rotations.createBasicXRotationMatrix(Math.PI/180*-142);
		// Combine to single matrix
		SimpleMatrix combined = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(rotZ, rotY), rotX);
		// Write matrix to stdout
		System.out.println(combined);
		// Create a new matrix with rotation about Y
		SimpleMatrix rotY2 = Rotations.createBasicRotationMatrix(BasicAxis.Y_AXIS, Math.PI/2);
		// output the matrix
		System.out.println(rotY2);
		// compute element-wise difference between the two matrices
		SimpleMatrix diff = SimpleOperators.subtract(combined, rotY2);
		// write the Frobenius norm to stdout:
		System.out.println("Frobenius norm of difference:"  + diff.norm(MatrixNormType.MAT_NORM_FROBENIUS));
		// Here we found a singularity of the Euler description. Both matrices are almost identical (up to numerical accucary).
		
		SimpleMatrix test = new SimpleMatrix("[[1, 0, 0]; [0, 0, 1]; [0,-1,0];]");
		
		
		System.out.println(SimpleOperators.multiplyMatrixProd(test.transposed(),combined));
	}

}

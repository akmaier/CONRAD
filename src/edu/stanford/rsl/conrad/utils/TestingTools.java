package edu.stanford.rsl.conrad.utils;


import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.Rotations.BasicAxis;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import static org.junit.Assert.assertTrue;


public abstract class TestingTools {

	///////////////////////////////////////////////////////
	// numerical equivalence                             //
	///////////////////////////////////////////////////////
	
	/** delta for error margins */
	public static final double DELTA = Math.sqrt(edu.stanford.rsl.conrad.utils.CONRAD.DOUBLE_EPSILON);
	
	/** own assert for matrices */
	public static final void assertEqualElementWise(final SimpleMatrix M1, final SimpleMatrix M2, final double delta) {
		assertTrue(SimpleOperators.equalElementWise(M1, M2, delta));
	}
	
	/** own assert for vectors */
	public static final void assertEqualElementWise(final SimpleVector v1, final SimpleVector v2, final double delta) {
		assertTrue(SimpleOperators.equalElementWise(v1, v2, delta));
	}
	
	
	///////////////////////////////////////////////////////
	// some generally useful random number generators    //
	///////////////////////////////////////////////////////

	/** Randomly generates either +1.0 or -1.0, i.e. from the set {-1, +1}. */
	public static final double randPmOne() { // {-1, +1}
		return (Math.random() < 0.5) ? -1.0 : 1.0;
	}
	
	/** Randomly generates a number in [-2.0, 2.0) but not +1.0 or -1.0, i.e. from the set [-2.0, 2.0) \ {-1.0, 1.0}. */
	public static final double randNotPmOne() { // [-2, 2) \ {-1, +1}
		double notPmOne;
		do notPmOne = 4.0 * (Math.random() - 0.5);
		while (Math.abs(Math.abs(notPmOne) - 1.0) < DELTA);
		return notPmOne;
	}

	/** Randomly generates a non-negative number, i.e. from the set [0.0, 1.0). */
	public static final double randNonNegative() { // [0, 1), with an increased probability (50%) for 0
		return (Math.random() < 0.5) ? 0.0 : Math.random();
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final double randPositive() { // (0, 1)
		return Math.random() + Double.MIN_VALUE;
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final double randNonPositive() { // (-1, 0], with an increased probability (50%) for 0
		return (Math.random() < 0.5) ? 0.0 : -Math.random();
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final double randNegative() { // (-1, 0)
		return -randPositive();
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final double randNonZero() { // (-1, 1) \ {0}
		return randPmOne() * randPositive();
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final double rand(final double min, final double max) { //  [min, max)
		return min + (max-min)*Math.random();
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final int rand(final int min, final int max) { //  {min, ..., max}
		return (int)Math.floor(rand((double)min, (double)(max+1)));
	}
	
	/** Randomly generates , i.e. from the set .*/
	public static final double randAng() { // [-pi, pi), with an increased probability for -pi, -pi/2, 0, pi/2
		double whatToDo = Math.random();
		final double oneFifth = 1.0/5.0;
		if (whatToDo < 1*oneFifth) return -Math.PI;
		else if (whatToDo < 2*oneFifth) return -0.5*Math.PI;
		else if (whatToDo < 3*oneFifth) return 0.0;
		else if (whatToDo < 4*oneFifth) return 0.5*Math.PI;
		else return (Math.random()-0.5) * 2.0 * Math.PI; // [-pi, pi)
	}
	
	/** Randomly generates a vector of the given length. */
	public static final SimpleVector randVector(int len) {
		assert len > 0;
		SimpleVector v = new SimpleVector(len);
		v.randomize(-1.0, 1.0);
		return v;
	}

	/** Randomly generates a matrix of the given size. */
	public static final SimpleMatrix randMatrix(int rows, int cols) {
		assert (rows > 0 && cols > 0);
		SimpleMatrix M = new SimpleMatrix(rows, cols);
		M.randomize(-1.0, 1.0);
		return M;
	}

	/** Randomly generates a matrix of the given size which is not singular. */
	public static final SimpleMatrix randMatrixNonSingular(int size) {
		assert (size > 0);
		SimpleMatrix M = new SimpleMatrix(size, size);
		do {
			M.randomize(-1.0, 1.0);
		} while (M.isSingular(DELTA));
		return M;
	}

	/** Randomly generates a 2x2 rotation matrix (representing a 2D rotation). */
	public static final SimpleMatrix randRotationMatrix2D() {
		final SimpleMatrix R = new SimpleMatrix(2, 2);
		final double ang = randAng();
		final double c = Math.cos(ang);
		final double s = Math.sin(ang);
		R.setElementValue(0, 0, c);
		R.setElementValue(0, 1, -s);
		R.setElementValue(1, 0, s);
		R.setElementValue(1, 1, c);
		return R;
	}

	/** Randomly generates a 3x3 rotation matrix (representing a 3D rotation). */
	public static final SimpleMatrix randRotationMatrix3D() {
		final SimpleMatrix Rx = Rotations.createBasicRotationMatrix(BasicAxis.X_AXIS, randAng());
		final SimpleMatrix Ry = Rotations.createBasicRotationMatrix(BasicAxis.Y_AXIS, randAng());
		final SimpleMatrix Rz = Rotations.createBasicRotationMatrix(BasicAxis.Z_AXIS, randAng());
		return SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(Rx, Ry), Rz);
	}

	/** Randomly generates an orthogonal matrix of the given size, i.e. a matrix from O(size). */
	public static final SimpleMatrix randOrthogonalMatrix(int size) {
		// create random O
		final SimpleMatrix Q = new SimpleMatrix(size, size);
		Q.randomize(-1.0, 1.0);
		// orthogonalize columns using the Gram-Schmidt algorithm
		for (int col = 0; col < size; ++col) {
			SimpleVector colVec = Q.getCol(col);
			for (int prevCol = 0; prevCol < col; ++prevCol) {
				SimpleVector prevColVec = Q.getCol(prevCol);
				colVec.subtract(prevColVec.multipliedBy(SimpleOperators.multiplyInnerProd(colVec, prevColVec)));
			}
			colVec.normalizeL2();
			Q.setColValue(col, colVec);
		}

		// orthogonalize rows using the Gram-Schmidt algorithm
		// this additional orthogonalization is not necessary in theory but should enhance the numerical orthogonality of the matrix
		for (int row = 0; row < size; ++row) {
			SimpleVector rowVec = Q.getRow(row);
			for (int prevRow = 0; prevRow < row; ++prevRow) {
				SimpleVector prevRowVec = Q.getRow(prevRow);
				rowVec.subtract(prevRowVec.multipliedBy(SimpleOperators.multiplyInnerProd(rowVec, prevRowVec)));
			}
			rowVec.normalizeL2();
			Q.setRowValue(row, rowVec);
		}

		return Q;
	}

	/** Randomly generates an special orthogonal matrix of the given size, i.e. a matrix from SO(size). */
	public static final SimpleMatrix randSpecialOrthogonalMatrix(int size) {
		// create orthogonal matrix
		final SimpleMatrix Q = randOrthogonalMatrix(size);
		
		// make sure it's from SO(size), not just O(size)
		Q.multiplyBy(Q.determinant());
		
		return Q;
	}

	/** Randomly generates an upper-triangular matrix of the given size. */
	public static final SimpleMatrix randUpperTriangularMatrix(int rows, int cols) {
		SimpleMatrix U = new SimpleMatrix(rows, cols);
		U.randomize(-1.0, 1.0);
		for (int row = 1; row < rows; ++row)
			for (int col = 0; col < Math.min(cols, row); ++col)
				U.setElementValue(row, col, 0.0);
		return U;
	}

	/** Randomly generates a lower-triangular matrix of the given size. */
	public static final SimpleMatrix randLowerTriangularMatrix(int rows, int cols) {
		SimpleMatrix L = new SimpleMatrix(rows, cols);
		L.randomize(-1.0, 1.0);
		for (int row = 0; row < rows; ++row)
			for (int col = row+1; col < cols; ++col)
				L.setElementValue(row, col, 0.0);
		return L;
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
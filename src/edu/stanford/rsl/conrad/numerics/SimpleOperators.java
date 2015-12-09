package edu.stanford.rsl.conrad.numerics;



/**
 * @author Andreas Keil
 */
public abstract class SimpleOperators {


	// **************************************************************** //
	// ******************* Vector/Vector operators ******************** //
	// **************************************************************** //

	/**
	 * <p>Computes the sum of supplied vectors
	 * <p> e.g. SimpleVector x = new SimpleVector(1,2,3);<br/>SimpleVector y = new SimpleVector(1,2,3);<br/> SimpleVector z = new SimpleVector(1,0,0);<br/>  SimpleOperators.add(x,y,z) returns [3,4,6];
	 * @param addends is comma-separated list or an array of vectors.
	 * @return the sum of supplied vectors.
	 */
	public static SimpleVector add(final SimpleVector... addends) {
		assert (addends.length >= 1) : new IllegalArgumentException("Need at least one argument for addition!");
		final SimpleVector result = addends[0].clone();
		for (int i = 1; i < addends.length; ++i)
			result.add(addends[i]);
		return result;
	}

	/**
	 * <p>subtracts v2 from v1
	 */
	public static SimpleVector subtract(final SimpleVector v1, final SimpleVector v2) {
		assert (v1.getLen() == v2.getLen()) : new IllegalArgumentException("Vector lengths must match for summation!");
		final SimpleVector result = new SimpleVector(v1.getLen());
		for (int i = 0; i < v1.getLen(); ++i)
			result.setElementValue(i, v1.getElement(i) - v2.getElement(i));
		return result;
	}
	
	/**
	 * <p>Multiplies the supplied vectors element wise
	 * <p> e.g. SimpleVector x = new SimpleVector(1,2,3);<br/>SimpleVector y = new SimpleVector(1,2,3);<br/> SimpleVector z = new SimpleVector(1,0,0);<br/>  SimpleOperators.multiplyElementWise(x,y,z) returns [1,0,0];
	 * @param factors is a comma-separated list or an array of vectors.
	 * @return element wise multiplication for supplied vectors
	 */
	public static SimpleVector multiplyElementWise(final SimpleVector... factors) {
		final SimpleVector result = factors[0].clone();
		for (int i = 1; i < factors.length; ++i)
			result.multiplyElementWiseBy(factors[i]);
		return result;
	}
	
	/**
	 * <p>Computes the element wise division of v1 by v2.
	 * <p> e.g. SimpleVector x = new SimpleVector(1,2,3);<br/>SimpleVector y = new SimpleVector(2,10,5);<br/> SimpleOperations.divideElementWise(x,y) returns [0.5,0.2,0.6];
	 * @param v1 is vector to be divided
	 * @param v2 is divisor
	 * @return a new vector containing the element-wise devision.
	 */
	public static SimpleVector divideElementWise(final SimpleVector v1, final SimpleVector v2) {
		final SimpleVector result = v1.clone();
		result.divideElementWiseBy(v2);
		return result;
	}
	
	/**
	 * <p> Computes the inner product multiplication (dot product) of v1 and v2.
	 * <p> SimpleVector x = new SimpleVector(1,2);<br/>SimpleVector y = new SimpleVector(3,10);<br/> SimpleOperations.multiplyInnerProd(x,y) = 1*3 + 2*10 = 23;
	 * @param v1 is first vector
	 * @param v2 is second vector
	 * @return the inner product multiplication of v1 and v2
	 */
	public static double multiplyInnerProd(final SimpleVector v1, final SimpleVector v2) {
		assert (v1.getLen() == v2.getLen()) : new IllegalArgumentException("Vector lengths must match for inner product calculation!");
		double result = 0.0;
		for (int i = 0; i < v1.getLen(); ++i)
			result += v1.getElement(i) * v2.getElement(i);
		return result;
	}
	
	/**
	 * Computes the outer product  multiplication of v1 and v2; i.e v1 x v2
	 * @param v1 is first vector
	 * @param v2 is second vector
	 * @return matrix representing v1 x v2
	 */
	public static SimpleMatrix multiplyOuterProd(final SimpleVector v1, final SimpleVector v2) {
		final SimpleMatrix result = new SimpleMatrix(v1.getLen(), v2.getLen());
		for (int r = 0; r < v1.getLen(); ++r)
			for (int c = 0; c < v2.getLen(); ++c)
				result.setElementValue(r, c, v1.getElement(r)*v2.getElement(c));
		return result;
	}

	/**
	 * Creates a new vector which is composed of all input vectors, stacked over each other.  
	 * @param parts  The vectors to concatenate.
	 * @return  The vertically concatenated vector.
	 */
	public static SimpleVector concatenateVertically(SimpleVector... parts) {
		final int noParts = parts.length;
		assert noParts >= 1 : new IllegalArgumentException("Supply at least one vector to concatenate!");
		final int[] lengths = new int[noParts];
		final int[] accumulatedLengthsBefore = new int[noParts];
		int totalLength = 0;
		for (int i = 0; i < noParts; ++i) {
			lengths[i] = parts[i].getLen();
			accumulatedLengthsBefore[i] = (i == 0) ? 0 : (accumulatedLengthsBefore[i-1] + lengths[i-1]);
			totalLength += lengths[i];
		}
		SimpleVector result = new SimpleVector(totalLength);
		for (int i = 0; i < noParts; ++i) result.setSubVecValue(accumulatedLengthsBefore[i], parts[i]);
		return result;
	}

	/**
	 * Creates a new matrix which is composed of all input column vectors, stacked next to each other.  
	 * @param columns  The vectors to stack.
	 * @return  The horizontally concatenated matrix.
	 */
	public static SimpleMatrix concatenateHorizontally(SimpleVector... columns) {
		final int cols = columns.length;
		assert cols >= 1 : new IllegalArgumentException("Supply at least one vector to concatenate!");
		final int rows = columns[0].getLen();
		assert rows >= 1 : new IllegalArgumentException("Vectors have to contain at least one element each!");
		SimpleMatrix result = new SimpleMatrix(rows, cols);
		for (int c = 0; c < cols; ++c)
			result.setColValue(c, columns[c]);
		return result;
	}

	public static boolean equalElementWise(final SimpleVector v1, final SimpleVector v2, final double delta) {
		if (v1.getLen() != v2.getLen()) throw new IllegalArgumentException("Vectors have different length!");
		for (int i = 0; i < v1.getLen(); ++i)
			if (Math.abs(v1.getElement(i) - v2.getElement(i)) > delta) return false;
		return true;
	}

	/**
	 * Computes and returns the element-wise maximum of all given vectors. 
	 * @param vectors  A comma-separated list or an array of vectors.
	 * @return  A new vector with the element-wise maximums of all given input vectors.
	 */
	public static SimpleVector max(final SimpleVector... vectors) {
		assert vectors.length >= 1 : new IllegalArgumentException("Provide at least one vector!");
		SimpleVector result = vectors[0].clone();
		final int rows = vectors[0].getLen();
		for (int i = 1; i < vectors.length; ++i) {
			assert vectors[i].getLen() == rows : new IllegalArgumentException("All vectors must have the same length!");
			for (int r = 0; r < rows; ++r)
				if (vectors[i].getElement(r) > result.getElement(r)) result.setElementValue(r, vectors[i].getElement(r));
		}
		return result;
	}

	/**
	 * Computes and returns the element-wise minimum of all given vectors. 
	 * @param vectors  A comma-separated list or an array of vectors.
	 * @return  A new vector with the element-wise minimums of all given input vectors.
	 */
	public static SimpleVector min(final SimpleVector... vectors) {
		assert vectors.length >= 1 : new IllegalArgumentException("Provide at least one vector!");
		SimpleVector result = vectors[0].clone();
		final int rows = vectors[0].getLen();
		for (int i = 1; i < vectors.length; ++i) {
			assert vectors[i].getLen() == rows : new IllegalArgumentException("All vectors must have the same length!");
			for (int r = 0; r < rows; ++r)
				if (vectors[i].getElement(r) < result.getElement(r)) result.setElementValue(r, vectors[i].getElement(r));
		}
		return result;
	}
	

	// **************************************************************** //
	// ******************* Matrix/Matrix operators ******************** //
	// **************************************************************** //
	/**
	 * Computes the sum of provided matrices
	 * @param addends  A comma-separated list or an array of matrices.
	 * @return a matrix representing the sum of provided matrices
	 */
	public static SimpleMatrix add(final SimpleMatrix... addends) {
		assert (addends.length >= 1) : new IllegalArgumentException("Need at least one argument for addition!");
		final SimpleMatrix result = addends[0].clone();
		for (int i = 1; i < addends.length; ++i)
			result.add(addends[i]);
		return result;
	}
	
	/**
	 * Subtracts M2 from M1
	 * @param M1 
	 * @param M2
	 * @return matrix representing the subtraction of M2 from M1
	 */
	public static SimpleMatrix subtract(final SimpleMatrix M1, final SimpleMatrix M2) {
		assert (M1.getRows() == M2.getRows() && M1.getCols() == M2.getCols()) : new IllegalArgumentException("Matrix sizes must match for summation!");
		final SimpleMatrix result = new SimpleMatrix(M1.getRows(), M1.getCols());
		for (int r = 0; r < result.getRows(); ++r)
			for (int c = 0; c < result.getCols(); ++c)
				result.setElementValue(r, c, M1.getElement(r, c) - M2.getElement(r, c));
		return result;
	}
	
	/**
	 * Computes the product of two matrices
	 * @param M1 is left matrix
	 * @param M2 is right matrix
	 * @return a matrix representing the product of provided matrices
	 */
	public static SimpleMatrix multiplyMatrixProd(final SimpleMatrix M1, final SimpleMatrix M2) {
		assert (M1.getCols() == M2.getRows()) : new IllegalArgumentException("Matrices' column/row dimensions must match for multiplication!");
		final SimpleMatrix result = new SimpleMatrix(M1.getRows(), M2.getCols());
		for (int r = 0; r < M1.getRows(); ++r)
			for (int c = 0; c < M2.getCols(); ++c)
				result.setElementValue(r, c, SimpleOperators.multiplyInnerProd(M1.getRow(r), M2.getCol(c)));
		return result;
	}
	

	public static SimpleMatrix multiplyElementWise(final SimpleMatrix... factors) {
		final SimpleMatrix result = factors[0].clone();
		for (int i = 1; i < factors.length; ++i)
			result.multiplyElementWiseBy(factors[i]);
		return result;
	}

	public static SimpleMatrix divideElementWise(final SimpleMatrix M1, final SimpleMatrix M2) {
		assert ((M1.getRows() == M2.getRows()) && (M1.getCols() == M2.getCols())) : new IllegalArgumentException("Matrices' dimensions must match for multiplication!");
		final SimpleMatrix result = new SimpleMatrix(M1.getRows(), M1.getCols());
		for (int r = 0; r < M1.getRows(); ++r)
			for (int c = 0; c < M1.getCols(); ++c)
				result.setElementValue(r, c, M1.getElement(r, c) / M2.getElement(r, c));
		return result;
	}

	public static boolean equalElementWise(final SimpleMatrix M1, final SimpleMatrix M2, final double delta) {
		if (M1.getRows() != M2.getRows()) throw new IllegalArgumentException("Matrices have different number of rows!");
		if (M1.getCols() != M2.getCols()) throw new IllegalArgumentException("Matrices have different number of columns!");
		for (int r = 0; r < M1.getRows(); ++r)
			for (int c = 0; c < M1.getCols(); ++c)
				if (Math.abs(M1.getElement(r, c) - M2.getElement(r, c)) > delta) return false;
		return true;
	}

	/**
	 * Computes and returns the element-wise maximum of all given matrices. 
	 * @param matrices  A comma-separated list or an array of matrices.
	 * @return  A new matrix with the element-wise maximums of all given input matrices.
	 */
	public static SimpleMatrix max(final SimpleMatrix... matrices) {
		assert matrices.length >= 1 : new IllegalArgumentException("Provide at least one vector!");
		SimpleMatrix result = matrices[0].clone();
		final int rows = matrices[0].getRows();
		final int cols = matrices[0].getCols();
		for (int i = 1; i < matrices.length; ++i) {
			assert (matrices[i].getRows() == rows) && (matrices[i].getCols() == cols) : new IllegalArgumentException("All matrices must have the same size!");
			for (int r = 0; r < rows; ++r)
				for (int c = 0; c < cols; ++c)
					if (matrices[i].getElement(r, c) > result.getElement(r, c)) result.setElementValue(r, c, matrices[i].getElement(r, c));
		}
		return result;
	}

	/**
	 * Computes and returns the element-wise minimum of all given matrices. 
	 * @param matrices  A comma-separated list or an array of matrices.
	 * @return  A new matrix with the element-wise minimums of all given input matrices.
	 */
	public static SimpleMatrix min(final SimpleMatrix... matrices) {
		assert matrices.length >= 1 : new IllegalArgumentException("Provide at least one vector!");
		SimpleMatrix result = matrices[0].clone();
		final int rows = matrices[0].getRows();
		final int cols = matrices[0].getCols();
		for (int i = 1; i < matrices.length; ++i) {
			assert (matrices[i].getRows() == rows) && (matrices[i].getCols() == cols) : new IllegalArgumentException("All matrices must have the same size!");
			for (int r = 0; r < rows; ++r)
				for (int c = 0; c < cols; ++c)
					if (matrices[i].getElement(r, c) < result.getElement(r, c)) result.setElementValue(r, c, matrices[i].getElement(r, c));
		}
		return result;
	}
	

	// **************************************************************** //
	// ******************* Matrix/Vector operators ******************** //
	// **************************************************************** //

	/**
	 * Performs a standard matrix-vector product.
	 * @param M  A matrix, used as first factor.
	 * @param v  A vector, used as second factor.
	 * @return  The matrix-vector product {@latex.inline $\\mathbf{M} \\cdot \\mathbf{v}$}.
	 */
	public static SimpleVector multiply(final SimpleMatrix M, final SimpleVector v) {
		assert (M.getCols() == v.getLen()) : new IllegalArgumentException("Matrix and Vector dimensions must match for multiplication!");
		final SimpleVector result = new SimpleVector(M.getRows());
		for (int r = 0; r < M.getRows(); ++r)
			result.setElementValue(r, SimpleOperators.multiplyInnerProd(M.getRow(r), v));
		return result;
	}

	/**
	 * Performs a vector-matrix product, assuming a row vector.
	 * @param v  A vector, assumed to be a row vector, used as first factor.
	 * @param M  A matrix, used as second factor.
	 * @return  The vector-matrix product {@latex.inline $\\mathbf{v}^T \\cdot \\mathbf{M}$}.
	 */
	public static SimpleVector multiply(final SimpleVector v, final SimpleMatrix M) {
		assert (v.getLen() == M.getRows()) : new IllegalArgumentException("Matrix and Vector dimensions must match for multiplication!");
		final SimpleVector result = new SimpleVector(M.getCols());
		for (int c = 0; c < M.getCols(); ++c)
			result.setElementValue(c, SimpleOperators.multiplyInnerProd(v, M.getCol(c)));
		return result;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
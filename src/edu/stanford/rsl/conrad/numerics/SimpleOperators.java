package edu.stanford.rsl.conrad.numerics;

import java.util.Collection;
import java.util.Iterator;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.utils.CONRAD;



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
	
	/**
	 * Performs an interpolation between two rigid transformations (rotation and translation) 
	 * matrices, represented as 4x4 affine matrices.
	 * 
	 * @param A First 3D Rigid transform matrix of size 4x4
	 * @param B Second 3D Rigid transform matrix of size 4x4
	 * @param weightA weight for first rigid transform
	 * @param weightB weight for second transform
	 * 
	 * @return interpolated rigid transform matrix of size 4x4
	 */
	public static SimpleMatrix interpolateRigidTransforms(SimpleMatrix A, SimpleMatrix B, double weightA, double weightB){
		assert(A.isRigidMotion3D(CONRAD.DOUBLE_EPSILON) && B.isRigidMotion3D(CONRAD.DOUBLE_EPSILON)) : new IllegalArgumentException("Matrix interpolation requires 3D rigid motion matrices (rotation + translatio only) of size 4x4!");
		
		boolean considerTranslationOnly = A.getSubMatrix(0, 0, 3, 3).isIdentity(CONRAD.DOUBLE_EPSILON);
		considerTranslationOnly &= B.getSubMatrix(0, 0, 3, 3).isIdentity(CONRAD.DOUBLE_EPSILON);
		
		// make sure weights add up to 1
		double sum = (weightA+weightB);
		weightA /= sum;
		weightB /= sum;
		
		SimpleMatrix outR = null;
		if(!considerTranslationOnly){
			SimpleMatrix firstRpart = A.getSubMatrix(0,0,3,3);
			SimpleMatrix scndRpart = B.getSubMatrix(0,0,3,3);
			SimpleMatrix firstInverseRpart = firstRpart.inverse(InversionType.INVERT_SVD);
			SimpleMatrix T = SimpleOperators.multiplyMatrixProd(firstInverseRpart,scndRpart);
			Matrix jamT = new Matrix(T.copyAsDoubleArray());
			EigenvalueDecomposition evd = jamT.eig();
			double[] real = evd.getRealEigenvalues();
			double[] imag = evd.getImagEigenvalues();
			for (int i = 0; i < 3; i++) {
				double angle = Math.atan2(imag[i], real[i]);
				angle*=weightB;
				real[i] = Math.cos(angle);
				imag[i] = Math.sin(angle);
			}
			Matrix newR = evd.getV().times(evd.getD()).times(evd.getV().inverse());
			outR = SimpleOperators.multiplyMatrixProd(firstRpart, new SimpleMatrix(newR.getArrayCopy()));
		}
		
		SimpleVector outCol = A.getSubCol(0, 3, 3).multipliedBy(weightA);
		outCol.add(B.getSubCol(0, 3, 3).multipliedBy(weightB));
		SimpleMatrix out = new SimpleMatrix(4,4);
		out.identity();
		out.setSubColValue(0, 3, outCol);
		if(!considerTranslationOnly){
			out.setSubMatrixValue(0, 0, outR);
		}
		return out;
	}
	
	/**
	 * Computes the mean rigid transform of a 3D transformation
	 * @param inputTransforms A collection of rigid transform matrices of size 4x4
	 * @return The mean rigid transform of the collections transform
	 */
	public static SimpleMatrix getMeanRigidTransform(Iterable<SimpleMatrix> inputTransforms){
		Iterator<SimpleMatrix> iter = inputTransforms.iterator();
		SimpleMatrix compareOut = SimpleMatrix.I_4.clone();
		int ctr = 0;
		while(iter.hasNext()){
			compareOut = SimpleOperators.interpolateRigidTransforms(
					iter.next(),
					compareOut,
					1, ctr);
			ctr++;
		}
		return compareOut;
	}
	
	/**
	 * method to compute the Pluecker dual coordinates of a vector L
	 * @param L: SimpleVector having 6 elements
	 * @return: SimpleMatrix of size 4x4
	 */
	public static SimpleMatrix getPlueckerMatrixDual(SimpleVector L) {
		
		SimpleMatrix L_out = new SimpleMatrix(4, 4);
		// first row
		L_out.setElementValue(0, 1, +L.getElement(5));
		L_out.setElementValue(0, 2, -L.getElement(4));
		L_out.setElementValue(0, 3, +L.getElement(3));
		
		// second row
		L_out.setElementValue(1, 0, -L.getElement(5));
		L_out.setElementValue(1, 2, +L.getElement(2));
		L_out.setElementValue(1, 3, -L.getElement(1));
		
		// third row
		L_out.setElementValue(2, 0, +L.getElement(4));
		L_out.setElementValue(2, 1, -L.getElement(2));
		L_out.setElementValue(2, 3, +L.getElement(0));
		
		// last row
		L_out.setElementValue(3, 0, -L.getElement(3));
		L_out.setElementValue(3, 1, +L.getElement(1));
		L_out.setElementValue(3, 2, -L.getElement(0));
		
		return L_out;
		
	}


	/**
	 * method to compute the Pluecker join of a line L and a point X
	 * @param L: line L as SimpleVector (6 entries)
	 * @param X: point X as SimpleVector (4 entries)
	 * @return: SimpleVector of size 4x1 representing a plane
	 */
	public static SimpleVector getPlueckerJoin(SimpleVector L, SimpleVector X) {
		
		//* calculate plane E (4x1) from [~L]x * X *//
		double v1 = + X.getElement(1)*L.getElement(5) - X.getElement(2)*L.getElement(4) + X.getElement(3)*L.getElement(3);
		double v2 = - X.getElement(0)*L.getElement(5) + X.getElement(2)*L.getElement(2) - X.getElement(3)*L.getElement(1);
		double v3 = + X.getElement(0)*L.getElement(4) - X.getElement(1)*L.getElement(2) + X.getElement(3)*L.getElement(0);
		double v4 = - X.getElement(0)*L.getElement(3) + X.getElement(1)*L.getElement(1) - X.getElement(2)*L.getElement(0);
		
		return new SimpleVector(v1, v2, v3, v4);
		
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
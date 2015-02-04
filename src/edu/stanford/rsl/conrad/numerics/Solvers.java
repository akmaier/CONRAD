package edu.stanford.rsl.conrad.numerics;


public abstract class Solvers {

	/**
	 * Solves the linear system of equations U*x = b with a square,
	 * upper-triangular matrix U using backward substitution.
	 * @param U  Square, upper-triangular, non-singular matrix.
	 * @param b  Right-hand side of the equation.
	 * @return  The solution vector x.
	 */
	public static SimpleVector solveUpperTriangular(final SimpleMatrix U, final SimpleVector b) {
		final int n = U.getRows();
		assert U.isSquare();
		assert U.isUpperTriangular();
		assert b.getLen() == U.getRows();
		SimpleVector x = new SimpleVector(n);
		for (int i = n-1; i >= 0; --i) {
			double sum = 0.0;
			for (int j = i+1; j < n; ++j)
				sum += U.getElement(i, j)*x.getElement(j);
			x.setElementValue(i, (b.getElement(i)-sum)/U.getElement(i, i));
		}
		return x;
	}
	
	/**
	 * Solves the linear system of equations L*x = b with a square,
	 * lower-triangular matrix L using forward substitution.
	 * @param L  Square, lower-triangular, non-singular matrix.
	 * @param b  Right-hand side of the equation.
	 * @return  The solution vector x.
	 */
	public static SimpleVector solveLowerTriangular(final SimpleMatrix L, final SimpleVector b) {
		final int n = L.getRows();
		assert L.isSquare();
		assert L.isUpperTriangular();
		assert b.getLen() == L.getRows();
		SimpleVector x = new SimpleVector(n);
		for (int i = 0; i < n; ++i) {
			double sum = 0.0;
			for (int j = 0; j < i; ++j)
				sum += L.getElement(i, j)*x.getElement(j);
			x.setElementValue(i, (b.getElement(i)-sum)/L.getElement(i, i));
		}
		return x;
	}
	
	/**
	 * Solves the linear system of equations
	 * {@latex.inline $\\mathbf{A} \\cdot \\mathbf{x} = \\mathbf{b}$}
	 * with a square matrix A.
	 * @param A  Square, non-singular matrix.
	 * @param b  Vector of matching dimension.
	 * @return  The solution x to {@latex.inline $\\mathbf{A} \\cdot \\mathbf{x} = \\mathbf{b}$}.
	 */
	public static SimpleVector solveLinearSysytemOfEquations(final SimpleMatrix A, final SimpleVector b) {
		// TODO: Switch to LU decomposition once it's available.
		DecompositionQR qr = new DecompositionQR(A);
		return qr.solve(b);
	}

	/**
	 * Solves the linear least squares problem
	 * {@latex.inline $\\min_{\\mathbf{x}} \\| \\mathbf{A} \\cdot \\mathbf{x} - \\mathbf{b} \\|^2$}
	 * with a matrix A (with as least as much rows as columns).
	 * @param A  Square, "standing" matrix.
	 * @param b  Vector of matching dimension.
	 * @return  The optimal solution {@latex.inline $\\mathbf{x}^*$} minimizing
	 * {@latex.inline $\\| \\mathbf{A} \\cdot \\mathbf{x} - \\mathbf{b} \\|^2$}.
	 */
	public static SimpleVector solveLinearLeastSquares(final SimpleMatrix A, final SimpleVector b) {
		DecompositionQR qr = new DecompositionQR(A);
		return qr.solve(b);
	}

}

/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.numerics;

import Jama.Matrix;
import Jama.util.*;

/** Singular Value Decomposition.
   <P>
   For an m-by-n matrix A, the singular value decomposition is
   an m-by-(m or n) orthogonal matrix U, a (m or n)-by-n diagonal matrix S, and
   an n-by-n orthogonal matrix V so that A = U*S*V'.
   <P>
   The singular values, sigma[k] = S[k][k], are ordered so that
   sigma[0] >= sigma[1] >= ... >= sigma[n-1].
   <P>
   The singular value decompostion always exists, so the constructor will
   never fail.  The matrix condition number and the effective numerical
   rank can be computed from this decomposition.   
   <P>
   This class is mainly based on SingularValueDecomposition class in Jama in which 
   SVD sometimes fails on cases m < n. The bug has been fixed by Ron Boisvert <boisvert@nist.gov>. 
   Details of what were fixed can be found below:
   http://cio.nist.gov/esd/emaildir/lists/jama/msg01431.html
   http://cio.nist.gov/esd/emaildir/lists/jama/msg01430.html
   http://metamerist.blogspot.com/2008/04/svd-for-vertically-challenged.html

   Then, small changes were made to be compatible with CONRAD. 
   @author Jang-Hwan Choi
 */

public class DecompositionSVD implements java.io.Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 201880439875521920L;

	/* ------------------------
   Class variables
	 * ------------------------ */

	/** Arrays for internal storage of U and V.
   @serial internal storage of U.
   @serial internal storage of V.
	 */
	private double[][] U, V;

	/** Array for internal storage of singular values.
   @serial internal storage of singular values.
	 */
	private double[] s;

	/** Row and column dimensions.
   @serial row dimension.
   @serial column dimension.
   @serial U column dimension.
	 */
	private int m, n, ncu;

	/** Column specification of matrix U
   @serial U column dimension toggle
	 */

	private boolean thin;


	/** 
	 * <b>Old Constructor</b><br>
	 * Construct the singular value decomposition
	 * Structure to access U, S and V.
	 * @param Arg  Rectangular matrix
	 * 
	 */

	public DecompositionSVD (SimpleMatrix Arg) {
		this(Arg,true,true,true);
	}

	/* ------------------------
   Constructor
	 * ------------------------ */

	/** 
	 * Construct the singular value decomposition, i.e. a
	 * structure to access U, S and V.
	 * @param Arg   Rectangular matrix
	 * @param thin  If true U is economy sized
	 * @param wantu If true generate the U matrix
	 * @param wantv If true generate the V matrix
	 * 
	 */

	public DecompositionSVD (SimpleMatrix Arg, boolean thin, boolean wantu,
			boolean wantv) {

		// Derived from LINPACK code.
		// Initialize.
		double[][] A = Arg.copyAsDoubleArray();
		m = Arg.getRows();
		n = Arg.getCols();
		this.thin = thin;

		ncu = thin?Math.min(m,n):m;
		s = new double [Math.min(m+1,n)];
		if (wantu) U = new double [m][ncu];
		if (wantv) V = new double [n][n];
		double[] e = new double [n];
		double[] work = new double [m];

		// Reduce A to bidiagonal form, storing the diagonal elements
		// in s and the super-diagonal elements in e.

		int nct = Math.min(m-1,n);
		int nrt = Math.max(0,Math.min(n-2,m));
		int lu = Math.max(nct,nrt);
		for (int k = 0; k < lu; k++) {
			if (k < nct) {

				// Compute the transformation for the k-th column and
				// place the k-th diagonal in s[k].
				// Compute 2-norm of k-th column without under/overflow.
				s[k] = 0;
				for (int i = k; i < m; i++) {
					s[k] = Maths.hypot(s[k],A[i][k]);
				}
				if (s[k] != 0.0) {
					if (A[k][k] < 0.0) {
						s[k] = -s[k];
					}
					for (int i = k; i < m; i++) {
						A[i][k] /= s[k];
					}
					A[k][k] += 1.0;
				}
				s[k] = -s[k];
			}
			for (int j = k+1; j < n; j++) {
				if ((k < nct) & (s[k] != 0.0))  {

					// Apply the transformation.

					double t = 0;
					for (int i = k; i < m; i++) {
						t += A[i][k]*A[i][j];
					}
					t = -t/A[k][k];
					for (int i = k; i < m; i++) {
						A[i][j] += t*A[i][k];
					}
				}

				// Place the k-th row of A into e for the
				// subsequent calculation of the row transformation.

				e[j] = A[k][j];
			}
			if (wantu & (k < nct)) {

				// Place the transformation in U for subsequent back
				// multiplication.

				for (int i = k; i < m; i++) {
					U[i][k] = A[i][k];
				}
			}
			if (k < nrt) {

				// Compute the k-th row transformation and place the
				// k-th super-diagonal in e[k].
				// Compute 2-norm without under/overflow.
				e[k] = 0;
				for (int i = k+1; i < n; i++) {
					e[k] = Maths.hypot(e[k],e[i]);
				}
				if (e[k] != 0.0) {
					if (e[k+1] < 0.0) {
						e[k] = -e[k];
					}
					for (int i = k+1; i < n; i++) {
						e[i] /= e[k];
					}
					e[k+1] += 1.0;
				}
				e[k] = -e[k];
				if ((k+1 < m) & (e[k] != 0.0)) {

					// Apply the transformation.

					for (int i = k+1; i < m; i++) {
						work[i] = 0.0;
					}
					for (int j = k+1; j < n; j++) {
						for (int i = k+1; i < m; i++) {
							work[i] += e[j]*A[i][j];
						}
					}
					for (int j = k+1; j < n; j++) {
						double t = -e[j]/e[k+1];
						for (int i = k+1; i < m; i++) {
							A[i][j] += t*work[i];
						}
					}
				}
				if (wantv) {

					// Place the transformation in V for subsequent
					// back multiplication.

					for (int i = k+1; i < n; i++) {
						V[i][k] = e[i];
					}
				}
			}
		}

		// Set up the final bidiagonal matrix or order p.
		int p = Math.min(n,m+1);
		if (nct < n) {
			s[nct] = A[nct][nct];
		}
		if (m < p) {
			s[p-1] = 0.0;
		}
		if (nrt+1 < p) {
			e[nrt] = A[nrt][p-1];
		}
		e[p-1] = 0.0;

		// If required, generate U.
		if (wantu) {
			for (int j = nct; j < ncu; j++) {
				for (int i = 0; i < m; i++) {
					U[i][j] = 0.0;
				}
				U[j][j] = 1.0;
			}
			for (int k = nct-1; k >= 0; k--) {
				if (s[k] != 0.0) {
					for (int j = k+1; j < ncu; j++) {
						double t = 0;
						for (int i = k; i < m; i++) {
							t += U[i][k]*U[i][j];
						}
						t = -t/U[k][k];
						for (int i = k; i < m; i++) {
							U[i][j] += t*U[i][k];
						}
					}
					for (int i = k; i < m; i++ ) {
						U[i][k] = -U[i][k];
					}
					U[k][k] += 1.0;
					for (int i = 0; i < k-1; i++) {
						U[i][k] = 0.0;
					}
				} else {
					for (int i = 0; i < m; i++) {
						U[i][k] = 0.0;
					}
					U[k][k] = 1.0;
				}
			}
		}

		// If required, generate V.
		if (wantv) {
			for (int k = n-1; k >= 0; k--) {
				if ((k < nrt) & (e[k] != 0.0)) {
					for (int j = k+1; j < n; j++) {
						double t = 0;
						for (int i = k+1; i < n; i++) {
							t += V[i][k]*V[i][j];
						}
						t = -t/V[k+1][k];
						for (int i = k+1; i < n; i++) {
							V[i][j] += t*V[i][k];
						}
					}
				}
				for (int i = 0; i < n; i++) {
					V[i][k] = 0.0;
				}
				V[k][k] = 1.0;
			}
		}

		// Main iteration loop for the singular values.

		int pp = p-1;
		int iter = 0;
		double eps = Math.pow(2.0,-52.0);
		double tiny = Math.pow(2.0,-966.0);
		while (p > 0) {
			int k,kase;

			// Here is where a test for too many iterations would go.

			// This section of the program inspects for
			// negligible elements in the s and e arrays.  On
			// completion the variables kase and k are set as follows.

			// kase = 1     if s(p) and e[k-1] are negligible and k<p
			// kase = 2     if s(k) is negligible and k<p
			// kase = 3     if e[k-1] is negligible, k<p, and
			//              s(k), ..., s(p) are not negligible (qr step).
			// kase = 4     if e(p-1) is negligible (convergence).

			for (k = p-2; k >= -1; k--) {
				if (k == -1) {
					break;
				}
				if (Math.abs(e[k]) <=
					tiny + eps*(Math.abs(s[k]) + Math.abs(s[k+1]))) {
					e[k] = 0.0;
					break;
				}
			}
			if (k == p-2) {
				kase = 4;
			} else {
				int ks;
				for (ks = p-1; ks >= k; ks--) {
					if (ks == k) {
						break;
					}
					double t = (ks != p ? Math.abs(e[ks]) : 0.) + 
					(ks != k+1 ? Math.abs(e[ks-1]) : 0.);
					if (Math.abs(s[ks]) <= tiny + eps*t)  {
						s[ks] = 0.0;
						break;
					}
				}
				if (ks == k) {
					kase = 3;
				} else if (ks == p-1) {
					kase = 1;
				} else {
					kase = 2;
					k = ks;
				}
			}
			k++;

			// Perform the task indicated by kase.

			switch (kase) {

			// Deflate negligible s(p).

			case 1: {
				double f = e[p-2];
				e[p-2] = 0.0;
				for (int j = p-2; j >= k; j--) {
					double t = Maths.hypot(s[j],f);
					double cs = s[j]/t;
					double sn = f/t;
					s[j] = t;
					if (j != k) {
						f = -sn*e[j-1];
						e[j-1] = cs*e[j-1];
					}
					if (wantv) {
						for (int i = 0; i < n; i++) {
							t = cs*V[i][j] + sn*V[i][p-1];
							V[i][p-1] = -sn*V[i][j] + cs*V[i][p-1];
							V[i][j] = t;
						}
					}
				}
			}
			break;

			// Split at negligible s(k).

			case 2: {
				double f = e[k-1];
				e[k-1] = 0.0;
				for (int j = k; j < p; j++) {
					double t = Maths.hypot(s[j],f);
					double cs = s[j]/t;
					double sn = f/t;
					s[j] = t;
					f = -sn*e[j];
					e[j] = cs*e[j];
					if (wantu) {
						for (int i = 0; i < m; i++) {
							t = cs*U[i][j] + sn*U[i][k-1];
							U[i][k-1] = -sn*U[i][j] + cs*U[i][k-1];
							U[i][j] = t;
						}
					}
				}
			}
			break;

			// Perform one qr step.

			case 3: {

				// Calculate the shift.

				double scale = Math.max(Math.max(Math.max(Math.max(
						Math.abs(s[p-1]),Math.abs(s[p-2])),Math.abs(e[p-2])), 
						Math.abs(s[k])),Math.abs(e[k]));
				double sp = s[p-1]/scale;
				double spm1 = s[p-2]/scale;
				double epm1 = e[p-2]/scale;
				double sk = s[k]/scale;
				double ek = e[k]/scale;
				double b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1)/2.0;
				double c = (sp*epm1)*(sp*epm1);
				double shift = 0.0;
				if ((b != 0.0) | (c != 0.0)) {
					shift = Math.sqrt(b*b + c);
					if (b < 0.0) {
						shift = -shift;
					}
					shift = c/(b + shift);
				}
				double f = (sk + sp)*(sk - sp) + shift;
				double g = sk*ek;

				// Chase zeros.

				for (int j = k; j < p-1; j++) {
					double t = Maths.hypot(f,g);
					double cs = f/t;
					double sn = g/t;
					if (j != k) {
						e[j-1] = t;
					}
					f = cs*s[j] + sn*e[j];
					e[j] = cs*e[j] - sn*s[j];
					g = sn*s[j+1];
					s[j+1] = cs*s[j+1];
					if (wantv) {
						for (int i = 0; i < n; i++) {
							t = cs*V[i][j] + sn*V[i][j+1];
							V[i][j+1] = -sn*V[i][j] + cs*V[i][j+1];
							V[i][j] = t;
						}
					}
					t = Maths.hypot(f,g);
					cs = f/t;
					sn = g/t;
					s[j] = t;
					f = cs*e[j] + sn*s[j+1];
					s[j+1] = -sn*e[j] + cs*s[j+1];
					g = sn*e[j+1];
					e[j+1] = cs*e[j+1];
					if (wantu && (j < m-1)) {
						for (int i = 0; i < m; i++) {
							t = cs*U[i][j] + sn*U[i][j+1];
							U[i][j+1] = -sn*U[i][j] + cs*U[i][j+1];
							U[i][j] = t;
						}
					}
				}
				e[p-2] = f;
				iter++;
			}
			break;

			// Convergence.

			case 4: {

				// Make the singular values positive.

				if (s[k] <= 0.0) {
					s[k] = (s[k] < 0.0 ? -s[k] : 0.0);
					if (wantv) {
						for (int i = 0; i < n; i++) {
							V[i][k] = -V[i][k];
						}
					}
				}

				// Order the singular values.

				while (k < pp) {
					if (s[k] >= s[k+1]) {
						break;
					}
					double t = s[k];
					s[k] = s[k+1];
					s[k+1] = t;
					if (wantv && (k < n-1)) {
						for (int i = 0; i < n; i++) {
							t = V[i][k+1]; V[i][k+1] = V[i][k]; V[i][k] = t;
						}
					}
					if (wantu && (k < m-1)) {
						for (int i = 0; i < m; i++) {
							t = U[i][k+1]; U[i][k+1] = U[i][k]; U[i][k] = t;
						}
					}
					k++;
				}
				iter = 0;
				p--;
			}
			break;
			}
		}
		A = null;
	}

	/* ------------------------
   Public Methods
	 * ------------------------ */

	/** Return the left singular vectors
   @return     U
	 */

	public SimpleMatrix getU () {
		return U==null?null:(new SimpleMatrix(new Matrix(U,m,m>=n?(thin?Math.min(m+1,n):ncu):ncu)));
	}

	/** Return the right singular vectors
   @return     V
	 */

	public SimpleMatrix getV () {
		return V==null?null:new SimpleMatrix(new Matrix(V,n,n));
	}

	/** Return the one-dimensional array of singular values
   @return     diagonal of S.
	 */

	public double[] getSingularValues () {
		return s;
	}

	/** Return the diagonal matrix of singular values
   @return     S
	 */

	public SimpleMatrix getS () {
		SimpleMatrix X = new SimpleMatrix(new Matrix(m>=n?(thin?n:ncu):ncu,n));
		for (int i = Math.min(m,n)-1; i>=0; i--)
			X.setElementValue(i, i, s[i]);
		return X;
	}

	/** Return the diagonal matrix of the reciprocals of the singular values
   @return     S+
	 */

	public SimpleMatrix getreciprocalS () {
		SimpleMatrix X = new SimpleMatrix(new Matrix(n,m>=n?(thin?n:ncu):ncu));
		for (int i = Math.min(m,n)-1; i>=0; i--)
			X.setElementValue(i, i, s[i]==0.0?0.0:1.0/s[i]);
		return X;
	}

	/** Return the Moore-Penrose (generalized) inverse
	 *  Slightly modified version of Kim van der Linde's code
   @param omit if true tolerance based omitting of negligible singular values
   @return     A+
	 */

	public SimpleMatrix inverse(boolean omit) {
		double[][] inverse = new double[n][m];
		if(rank()> 0) {
			double[] reciprocalS = new double[s.length];
			if (omit) {
				double tol = Math.max(m,n)*s[0]*Math.pow(2.0,-52.0);
				for (int i = s.length-1;i>=0;i--)
					reciprocalS[i] = Math.abs(s[i])<tol?0.0:1.0/s[i];
			}
			else
				for (int i=s.length-1;i>=0;i--)
					reciprocalS[i] = s[i]==0.0?0.0:1.0/s[i];
			int min = Math.min(n, ncu);
			for (int i = n-1; i >= 0; i--)
				for (int j = m-1; j >= 0; j--)
					for (int k = min-1; k >= 0; k--)
						inverse[i][j] += V[i][k] * reciprocalS[k] * U[j][k];
		} 
		return new SimpleMatrix(new Matrix(inverse));
	}
	
	/** Return the Moore-Penrose inverse including Tikhonov regularization.
	 *  Reciprocal eigenvalues given as sigma / (sigma²+alpha), where alpha is the regularization parameter for A + alpha I.
	 *  @param omit if true tolerance based omitting of negligible singular values
	 *  @param alpha regularization parameter
	 *  @return     A+
	 *  @author Tobias Geimer
	 */
	public SimpleMatrix regularizedInverse(boolean omit, double alpha) {
		// Regularization parameter < 0.0 is invalid. Call non-regularized pseudo-inverse instead.
		if(alpha <= 0.0) return inverse(omit);
		
		double[][] inverse = new double[n][m];
		if(rank()> 0) {
			double[] reciprocalS = new double[s.length];
			if (omit) {
				double tol = Math.max(m,n)*s[0]*Math.pow(2.0,-52.0);
				for (int i = s.length-1;i>=0;i--){
					double filter = (s[i]) / (s[i]*s[i] + alpha);
					reciprocalS[i] = Math.abs(s[i])<tol?0.0:filter;
				}
			} else {
				for (int i=s.length-1;i>=0;i--) {
					double filter = (s[i]) / (s[i]*s[i] + alpha);
					reciprocalS[i] = s[i]==0.0?0.0:filter;
				}
			}
			int min = Math.min(n, ncu);
			for (int i = n-1; i >= 0; i--)
				for (int j = m-1; j >= 0; j--)
					for (int k = min-1; k >= 0; k--)
						inverse[i][j] += V[i][k] * reciprocalS[k] * U[j][k];
		} 
		return new SimpleMatrix(new Matrix(inverse));
	}

	/** Two norm
   @return     max(S)
	 */

	public double norm2 () {
		return s[0];
	}

	/** Two norm condition number
   @return     max(S)/min(S)
	 */

	public double cond () {
		return s[0]/s[Math.min(m,n)-1];
	}

	/** Effective numerical matrix rank
   @return     Number of nonnegligible singular values.
	 */

	public int rank () {
		double tol = Math.max(m,n)*s[0]*Math.pow(2.0,-52.0);
		int r = 0;
		for (int i = 0; i < s.length; i++) {
			if (s[i] > tol) {
				r++;
			}
		}
		return r;
	}
}

/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
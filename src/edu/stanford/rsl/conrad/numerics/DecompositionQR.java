package edu.stanford.rsl.conrad.numerics;



/**
 * Implements a QR decomposition for arbitrary matrices.
 *
 * This functor object allows to decompose any matrix \f$ \mathbf{A} \f$ into matrices 
 * \f$ \mathbf{Q} \f$ and \f$ \mathbf{R} \f$ such that
 * \f[
 *		\mathbf{A} = \mathbf{Q} \mathbf{R}.
 * \f]
 * Use it for solving systems of linear equations and for minimization problems
 * (with more equations than variables). The QR decomposition is not suited for
 * computing the set of solutions for an under-determined system of equations
 * (although it may be used to compute one of those solutions).
 *
 * Assuming \f$ \mathbf A \f$ is a \f$ m \times n \f$ matrix, the decomposition is
 * performed such that the matrix \f$ \mathbf Q \f$ has dimensions \f$ m \times m \f$ and
 * is orthogonal and the matrix \f$ \mathbf{R} \f$ has dimensions \f$ m \times n \f$ and
 * has upper triangular form (\f$ r_{i,j} = 0 \f$ for \f$ i>j \f$).
 * Internally, only a compressed version of the factors \f$ \mathbf{Q} \f$ and
 * \f$ \mathbf{R} \f$ is stored which together needs approximately as much space as the
 * original matrix \f$ \mathbf{A} \f$.
 *
 * Usage:
 * \code
 * Nude::Decomposition::QR<> qr;
 * qr(A);
 * qr.solve(b);
 * \endcode
 *
 * \sa RQ
 * \sa Deuflhard/Hohmann: Numerische Mathematik I
 * \sa The development of this code started out with the implementation of TNT/JAMA at
 *     http://math.nist.gov/tnt/. However, it was completely revised, optimized, and commented.
 *
 * \author Andreas Keil
 *
 * \todo Handle singularity / do pivoting when decomposing.
 * \todo Improve method isFullRank() to check against an epsilon.
 */
public class DecompositionQR {

	/**
	 * Constructor performing the actual decomposition of a matrix {@latex.inline $\\mathbf{A}$}
	 * and storing the result in an internal format.
	 *
	 * Decomposition is performed as soon as a Matrix is given to this
	 * constructor and the result is stored internally. Afterwards, the other
	 * other members ({@link #solve(SimpleVector b)}, {@link #solve(SimpleMatrix B)},
	 * {@link #getQ()}, and {@link #getR()}) can be used multiple times without having
	 * to recompute the decomposition.
	 *
	 * @param A  The Matrix to be decomposed.
	 */
	public DecompositionQR(final SimpleMatrix A) {
		// get dimensions
		final int m = A.getRows();
		final int n = A.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A

		// initialize internal variables
		this.QR = new SimpleMatrix(A);
		this.Rdiag = new SimpleVector(min_mn);

		// main loop over columns of A which are to be reduced to upper triangular form
		for (int k = 0; k < min_mn; ++k) {

			// compute 2-norm of k-th column without under/overflow using C++'s _hypot for computing the sqrt(a^2 + b^2)
			double alpha = 0.0;
			for (int i = k; i < m; ++i) alpha += this.QR.getElement(i, k) * this.QR.getElement(i, k);
			alpha = Math.sqrt(alpha);

			if (alpha != 0.0) { //TODO: error handling or permutation for singular case when alpha == 0.0?
				// choose the sign for alpha to prevent loss of significance
				if (this.QR.getElement(k, k) > 0.0) alpha = -alpha;

				// form k-th Householder vector v as (y/alpha - e_1), so that Qy = alpha e_1
				for (int i = k; i < m; ++i) this.QR.divideElementBy(i, k, alpha);
				this.QR.subtractFromElement(k, k, 1.0);

				// apply transformation to remaining columns x, where Qx = x + (v'*x / v_1) * v
				for (int j = k + 1; j < n; ++j) {
					// compute v'*x
					double s = 0.0;
					for (int i = k; i < m; ++i) s += this.QR.getElement(i, k) * this.QR.getElement(i, j);
					// compute v'*x / v_1
					s /= this.QR.getElement(k, k);
					// compute the final result Qx = x + (v'*x / v_1) * v
					for (int i = k; i < m; ++i) this.QR.addToElement(i, j, s*this.QR.getElement(i, k));
				}
			}
			// store diagonal entry of R
			this.Rdiag.setElementValue(k, alpha);
		}
	
	}

	/**
	 * Specifies whether the input Matrix {@latex.inline $A$} has full rank.
	 *
	 * @return Whether the input Matrix has full rank ({@latex.inline $\\min\\{m, n\\}$}).
	 */
	public boolean isFullRank() {
		final int min_mn = this.Rdiag.getLen();
		for (int l = 0; l < min_mn; ++l)
			if (this.Rdiag.getElement(l) == 0.0) return false;
		return true;
	}

	/**
	 * Computes solution Vector {@latex.inline $\\mathbf x$} for the right-hand-side {@latex.inline $\\mathbf b$}.
	 *
	 * Depending on the size of {@latex.inline $\\mathbf A \\in \\mathbb{R}^{m \\times n}$}, the problem task
	 * can either be the solution of a system of equations with a unique solution or a
	 * minimization task:
	 * task:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align}
	 *   \\mathbf A \\mathbf x = \\mathbf b &: m = n \\quad\\text{(unique solution)} \\\\
	 *   \\min_{\\mathbf x}\\| \\mathbf A \\mathbf x - \\mathbf b \\|_2 &: m > n \\quad\\text{(minimization)} \\\\
	 *   \\mathbf A \\mathbf x = \\mathbf b &: m < n \\quad\\text{(one of the infinite set of solutions)}
	 * \\end{align} }
	 * For {@latex.inline $m < n$} the QR decomposition computes one of the infinite number of
	 * solutions to the under-determined system of equations. Better use the RQ
	 * decomposition in this case.
	 *
	 * <p><em>Remark:</em> If you want to further improve the accuracy of your solution in case (1),
	 * perform a correction step in the following manner:<br>
	 * {@code
	 * x = qr.solve(b);
	 * x += qr.solve(b - A*x);
	 * }
	 *
	 * @param b  The right-hand-side Vector.
	 * @return  The solution Vector {@latex.inline $\\mathbf x$}.
	 *
	 * @see #solve(SimpleMatrix)
	 */
	public SimpleVector solve(final SimpleVector b) {
		final SimpleMatrix B = new SimpleMatrix(b.getLen(), 1);
		B.setColValue(0, b);
		final SimpleMatrix X = this.solve(B);
		return X.getCol(0);
	}

	/**
	 * Computes solution Matrix {@latex.inline $\\mathbf X$} for the right-hand-side {@latex.inline $\\mathbf B$}.
	 *
	 * This method does the same as {@link #solve(SimpleVector)} but for multiple
	 * right-hand-side vectors, given as a Matrix.
	 *
	 * @param B  The right-hand-side Matrix for which a solution is computed column-wise.
	 * @return  The solution Matrix {@latex.inline $\\mathbf X$}.
	 *
	 * @see #solve(SimpleVector)
	 */
	public SimpleMatrix solve(final SimpleMatrix B) {
		final int m = this.QR.getRows();
		final int n = this.QR.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A
		final int nb = B.getCols(); // column number of right-hand-side (number of RHSs to compute a solution for)

		// check for conformant arrays
		if (B.getRows() != m) throw new IllegalArgumentException("Number of rows of A and B do not conform!");

		// check whether we can actually use the QR decomposition for solving the given problem
		//if (n > m) ; //TODO Maybe issue a warning here, since for an under-determined system, QR can compute a solution candidate, but not the one with the smallest norm and not the whole solution space (as can RQ)

		// check for rank deficiency
		if (!isFullRank()) throw new IllegalArgumentException("Matrix does not have full rank!");

		// make internal copy of B for further inplace computations
		SimpleMatrix X = B.clone();

		// compute X = Q^T * B = Qmin_mn * ... * Q1 * B (inplace in X)
		for (int k = 0; k < min_mn; ++k) {
			// iterate through columns of B
			for (int j = 0; j < nb; ++j) {
				// compute v'*x
				double s = 0.0;
				for (int i = k; i < m; ++i) s += this.QR.getElement(i, k) * X.getElement(i, j);
				// compute v'*x / v_1
				s /= this.QR.getElement(k, k);
				// compute the final result Qx = x + (v'*x / v_1) * v
				for (int i = k; i < m; ++i) X.addToElement(i, j, s*this.QR.getElement(i,k));
			}
		}

		// solve R*X' = X (inplace in X)
		// iterate through columns of X to compute columns of X'
		for (int j = 0; j < nb; ++j) {
			// iterate through non-zero rows of R to compute non-zero rows of X'
			for (int k = min_mn-1; k >= 0; --k) {
				for (int i = k+1; i < min_mn; ++i) X.subtractFromElement(k, j, this.QR.getElement(k, i) * X.getElement(i, j));
				X.divideElementBy(k, j, this.Rdiag.getElement(k));
			}
		}

		// return solution (m == n), approximation (m > n), or a representative among the set of solutions (m < n)
		if (m > n) {
			X = X.getSubMatrix(0, 0, n, nb);
		} else if (m < n) {
			SimpleMatrix Xnew = new SimpleMatrix(n, nb);
			Xnew.setSubMatrixValue(0, 0, X);
			X = Xnew;
			for (int i = m+1; i < n; ++i)
				for (int j = 0; j < nb; ++j)
					X.setElementValue(i, j, 0.0);
		}
		
		return X;
	}

	/**
	 * Compute R from the internal storages QR and Rdiag.
	 *
	 * R has the same dimensions as the original matrix.
	 * An economy-sized version would only return the non-zero submatrix.
	 */
	public SimpleMatrix getR() {
		final int m = this.QR.getRows();
		final int n = this.QR.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A

		// construct R as zero matrix
		final SimpleMatrix R = new SimpleMatrix(m, n);

		// fill non-zero positions of R with Rdiag and upper-right part of QR
		for (int i = 0; i < min_mn; ++i) {
			R.setElementValue(i, i, this.Rdiag.getElement(i));
			for (int j = i+1; j < n; ++j) R.setElementValue(i, j, this.QR.getElement(i,j));
		}
		
		// return result
		return R;
	}

	/**
	 * Compute Q from the internal storage QR.
	 *
	 * Q is a square matrix with both dimensions equaling the row number of the original matrix.
	 * An economy-sized version would only return the columns of Q corresponding to the economy-sized version of R.
	 */
	public SimpleMatrix getQ() {
		final int m = this.QR.getRows();
		final int n = this.QR.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A

		// start with an identity matrix
		final SimpleMatrix Q = new SimpleMatrix(m, m);
		Q.identity();

		// iterate through Householder reflections to compute Q = Q1 * ... * Qmin_mn * I
		for (int k = min_mn - 1; k >= 0; --k) {
			if (this.QR.getElement(k, k) != 0.0) { // a zero entry in this position means we didn't have to reflect anything since we encountered a singularity => Q_k = I
				// iterate through columns x of Q
				for (int j = k; j < m; ++j) {
					// compute v'*x
					double s = 0.0;
					for (int i = k; i < m; ++i) s += this.QR.getElement(i, k) * Q.getElement(i, j);
					// compute v'*x / v_1
					s /= this.QR.getElement(k, k);
					// compute the final result Qx = x + (v'*x / v_1) * v
					for (int i = k; i < m; ++i) Q.addToElement(i, j, s*this.QR.getElement(i,k));
				}
			}
		}
		
		// return result
		return Q;
	}
	
	
	private final SimpleMatrix QR;
	private final SimpleVector Rdiag;

}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.numerics;


/**
 * Implements an RQ decomposition for arbitrary matrices.
 *
 * This functor object allows to decompose any matrix {@latex.inline $\\mathbf{A}$} into matrices 
 * {@latex.inline $\\mathbf{R}$} and {@latex.inline $\\mathbf{Q}$} such that
 * {@latex.ilb \\[
 *		\\mathbf{A} = \\mathbf{R} \\mathbf{Q}.
 * \\]}
 * Use it for solving systems of linear equations and for computing the infinite set
 * of solutions for an under-determined system of equations. The RQ decomposition is
 * not suited for minimization problems (with more equations than variables).
 *
 * Assuming {@latex.inline $\\mathbf{A}$} is a {@latex.inline $m \\times n$} matrix, the decomposition is
 * performed such that the matrix {@latex.inline $\\mathbf{R}$} has dimensions {@latex.inline $m \\times n$} and
 * has upper triangular form ({@latex.inline $r_{i,j} = 0$} for {@latex.inline $m-i < n-j$}) and the
 * matrix {@latex.inline $\\mathbf{Q}$} has dimensions {@latex.inline $n \\times n$} and is orthogonal.
 * Internally, only a compressed version of the factors {@latex.inline $\\mathbf{R}$} and
 * {@latex.inline $\\mathbf{Q}$} is stored which together needs approximately as much space as the
 * original matrix {@latex.inline $\\mathbf{A}$}.
 *
 * Usage:
 * {@code
 * Nude::Decomposition::RQ<> rq;
 * rq(A);
 * rq.solve(b);
 * }
 *
 * @see DecompositionQR
 * @see edu.stanford.rsl.conrad.geometry.Projection
 *
 * @author Andreas Keil (Implementations started from JAMA code but now contain strong modifications. Actually, now it's a mix of JAMA and the version in Deuflhard/Hohmann: Numerische Mathematik I.)
 */
public class DecompositionRQ {
	
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
	public DecompositionRQ(final SimpleMatrix A) {
		// get dimensions
		final int m = A.getRows();
		final int n = A.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A

		// initialize internal matrices
		this.RQ = new SimpleMatrix(A);
		this.Rdiag = new SimpleVector(m);

		// main loop over rows of A which are to be reduced to upper triangular form
		for (int k = m-1; k >= m-min_mn; --k) {

			// compute 2-norm of k-th row without under/overflow using C++'s _hypot for computing the sqrt(a^2 + b^2)
			double alpha = 0;
			for (int j = 0; j <= k+n-m; ++j) alpha += this.RQ.getElement(k,j)*this.RQ.getElement(k,j);
			alpha = Math.sqrt(alpha);

			if (alpha != 0.0) { //TODO: error handling or permutation for singular case when alpha == 0.0?
				// choose the sign for alpha to prevent loss of significance
				if (this.RQ.getElement(k, k+n-m) > 0.0) alpha = -alpha;

				// form k-th Householder vector v as (y/alpha - e_1), so that Qy = alpha e_1
				for (int j = 0; j <= n-m+k; ++j) this.RQ.divideElementBy(k, j, alpha);
				this.RQ.subtractFromElement(k, k+n-m, 1.0);

				// apply transformation to remaining rows x, where Qx = x + (v'*x / v_1) * v
				for (int i = k-1; i >= 0; --i)
				{
					// compute v'*x
					double s = 0.0;
					for (int j = 0; j <= k+n-m; ++j) s += this.RQ.getElement(k, j) * this.RQ.getElement(i, j);
					// compute v'*x / v_1
					s /= this.RQ.getElement(k, k + n - m);
					// compute the final result Qx = x + (v'*x / v_1) * v
					for (int j = 0; j <= k+n-m; ++j) this.RQ.addToElement(i, j, s*this.RQ.getElement(k, j));
				}
			}
			// store diagonal entry of R
			this.Rdiag.setElementValue(k+min_mn-m, alpha);
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
	 * Computes solution Vector {@latex.inline $\\mathbf{x}$} for the right-hand-side {@latex.inline $\\mathbf b$}.
	 *
	 * Depending on the size of {@latex.inline $\\mathbf A \\in \\mathbb{R}^{m \\times n}$}, the problem task
	 * can be either the solution of a system of equations with one or an infinite number
	 * of solutions:
	 * {@latex.ilb %preamble{\\usepackage{amsmath}} \\begin{align}
	 *   \\mathbf A \\mathbf x = \\mathbf b &: m = n \\quad\\text{(unique solution)} \\\\
	 *   \\min_{\\mathbf x}\\| \\mathbf A \\mathbf x - \\mathbf b \\|_2 &: m > n \\quad\\text{(RQ decomposition not suited!)} \\\\
	 *   \\mathbf A \\mathbf x = \\mathbf b &: m < n \\quad\\text{(infinite number of solutions)}
	 * \\end{align}}
	 * {@latex.inline $m > n$} throws an exception, since the RQ decomposition is not suited for
	 * solving minimization problems. Use a QR decomposition in this case. For {@latex.inline $m < n$}
	 * the RQ decomposition computes the minimum norm solution to the under-determined system of
	 * equations. The {@latex.inline $n-m$} dimensional set of all solutions can then be obtained by adding
	 * any linear combination of the first {@latex.inline $n-m$} rows of {@latex.inline $\\mathbf{Q}$}.
	 *
	 * <p><em>Remark:</em> If you want to further improve the accuracy of your solution in case (1),
	 * perform a correction step in the following manner:<br>
	 * {@code
	 * x = rq.solve(b);
	 * x += rq.solve(b - A*x);
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
	 * This method does the same as @link{#solve(SimpleVector)} but for multiple
	 * right-hand-side vectors, given as a Matrix.
	 *
	 * @param B The right-hand-side Matrix for which a solution is computed column-wise.
	 * @return The solution Matrix {@latex.inline $\\mathbf X$}.
	 *
	 * @see #solve(SimpleVector)
	 */
	public SimpleMatrix solve(final SimpleMatrix B) {
		final int m = this.RQ.getRows();
		final int n = this.RQ.getCols();
		final int nb = B.getCols(); // column number of right-hand-side (number of RHSs to compute a solution for)
	
		// check for conformant arrays
		if (B.getRows() != m) throw new IllegalArgumentException("Number of rows of A and B do not conform!");
	
		// check whether we can actually use the RQ decomposition for solving the given problem
		// Attention: Do not execute later code for m > n, since it is specialized to m <= n!
		if (m > n) throw new IllegalArgumentException("System is over-determined! Use QR decomposition for solving this!");
	
		// check for rank deficiency
		if (!this.isFullRank()) throw new IllegalArgumentException("Matrix does not have full rank!");
	
		// make internal copy of B for further inplace computations
		final SimpleMatrix X = new SimpleMatrix(n, nb);
		X.setSubMatrixValue(n-m, 0, B);
	
		// solve R*X'2 = B, where X'2 are the last m rows of X' which are not arbitrary (inplace in X)
		// iterate through columns of B to compute columns of X'
		for (int j = 0; j < nb; ++j)
		{
			// iterate through rows of R to compute non-free rows of X' (X'2)
			for (int i = m-1; i >= 0; --i)
			{
				for (int k = i+1; k < m; ++k)
				{
					X.subtractFromElement(i+n-m, j, this.RQ.getElement(i, k+n-m) * X.getElement(k+n-m, j));
				}
				X.divideElementBy(i+n-m, j, this.Rdiag.getElement(i));
			}
		}
	
		// compute X = Q^T * X' = Q1 * ... * Qm * X' (inplace in X)
		for (int k = 0; k < m; ++k)
		{
			if (this.RQ.getElement(k, k+n-m) != 0.0) // a zero entry in this position means we didn't have to reflect anything since we encountered a singularity => Q_k = I
			{
				// iterate through columns of B
				for (int j = 0; j < nb; ++j)
				{
					// compute v'*x
					double s = 0;
					for (int i = 0; i <= k+n-m; ++i)
						s += this.RQ.getElement(k, i) * X.getElement(i, j);
					// compute v'*x / v_1
					s /= this.RQ.getElement(k, k+n-m);
					// compute the final result Qx = x + (v'*x / v_1) * v
					for (int i = 0; i <= k+n-m; ++i)
						X.addToElement(i, j, s*this.RQ.getElement(k, i));
				}
			}
		}
		
		// return result
		return X;
	}
	
	/**
	 * Computes the upper-triangular {@latex.inline $m \\times n$} matrix {@latex.inline $\\mathbf{R}$}.
	 *
	 * The matrices {@latex.inline $\\mathbf{R}$} and {@latex.inline $\\mathbf{Q}$} are stored in an internal
	 * format and only explicitly computed on request. They are not needed for
	 * solving one of the equations mentioned in the solve() members! Only
	 * call this method if you really actually need the full decomposed matrices.
	 *
	 * @return The upper-triangular Matrix {@latex.inline $\\mathbf{R} \\in \\mathbb{R}^{m \\times n}$}.
	 *
	 * @see #getQ()
	 * @see #solve(SimpleVector)
	 * @see #solve(SimpleMatrix)
	 */
	public SimpleMatrix getR() {
		final int m = this.RQ.getRows();
		final int n = this.RQ.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A
		
		// construct R as zero matrix
		final SimpleMatrix R = new SimpleMatrix(m, n);
		
		// fill non-zero positions of R with Rdiag and upper-right part of RQ
		for (int j = n-min_mn; j < n; ++j) {
			R.setElementValue(j-n+m, j, this.Rdiag.getElement(j-n+min_mn));
			for (int i = 0; i < j-n+m; ++i) R.setElementValue(i, j, this.RQ.getElement(i, j));
		}
		
		// return result
		return R;
	}
	
	/**
	 * Computes the orthogonal {@latex.inline $n \\times n$} matrix {@latex.inline $\\mathbf{Q}$}.
	 *
	 * The matrices {@latex.inline $\\mathbf{Q}$} and {@latex.inline $\\mathbf{R}$} are stored in an internal
	 * format and only explicitly computed on request. They are not needed for
	 * solving one of the equations mentioned in the solve() members! Only
	 * call this method if you really actually need the full decomposed matrices.
	 *
	 * @return The orthogonal Matrix {@latex.inline $\\mathbf{Q} \\in \\mathbb{R}^{n \\times n}$}.
	 *
	 * @see #getR()
	 * @see #solve(SimpleVector)
	 * @see #solve(SimpleMatrix)
	 */
	public SimpleMatrix getQ() {
		final int m = this.RQ.getRows();
		final int n = this.RQ.getCols();
		final int min_mn = Math.min(m, n); // number of Householder reflections used to construct an upper triangular R from A

		// start with an identity matrix
		final SimpleMatrix Q = new SimpleMatrix(n, n);
		Q.identity();

		// iterate through Householder reflections to compute Q = I * Qmin_mn * ... * Q1
		for (int k = m-min_mn; k < m; ++k) {
			if (this.RQ.getElement(k, k+n-m) != 0.0) { // a zero entry in this position means we didn't have to reflect anything since we encountered a singularity => Q_k = I
				// iterate through rows x of Q
				for (int i = 0; i <= k+n-m; ++i) {
					// compute v'*x
					double s = 0;
					for (int j = 0; j <= k+n-m; ++j)
						s += this.RQ.getElement(k, j) * Q.getElement(i, j);
					// compute v'*x / v_1
					s /= this.RQ.getElement(k, k+n-m);
					// compute the final result Qx = x + (v'*x / v_1) * v
					for (int j = 0; j <= k+n-m; ++j) Q.addToElement(i, j, s*this.RQ.getElement(k, j));
				}
			}
		}
		
		// return result
		return Q;
	}

	
	private final SimpleMatrix RQ;
	private final SimpleVector Rdiag;
	
}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

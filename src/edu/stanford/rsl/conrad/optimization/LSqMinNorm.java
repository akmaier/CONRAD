package edu.stanford.rsl.conrad.optimization;

import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This class implements a solver for a linear system of equations with additional regularization of the solution vector favoring smaller norms.
 * The regularization is controlled using a Lagrange multiplier lambda. If lambda is set to 0, this class will produce a non-regularized least squares 
 * solution.
 * Solves: min_x |A * x - b|^2 + lambda * |x|^2,
 * which be reformulated into |A' * x - b'| using 
 * A' = (A , sqrt(lambda) * 1_n)^T, b' + (b , 0_n)^T
 * Uses a singular value decomposition to solve the final system of equations.
 * @author Mathias Unberath
 *
 */
public class LSqMinNorm {
	
	/**
	 * The coefficients multiplied to the columns of the matrix in the linear system of equations.
	 */
	private SimpleVector x;
	
	/**
	 * The right hand side of the system of equations.
	 */
	private SimpleVector b;
	
	/**
	 * The matrix containing the parameters.
	 */
	private SimpleMatrix a;
	
	/**
	 * Lagrange Multiplier for the minimal norm regularization.
	 */
	private double lambda = 0.005;
	
	/**
	 * Number of columns in the SimpleMatrix.
	 */
	private int nCol;
	
	/**
	 * Number of rows in the SimpleMatrix.
	 */
	private int nRow;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================

	/** 
	 * Constructs the solver object and calls the solving method on the lineaer system of equations A * x = b subject to |x|^2 -> min.
	 * @param a The matrix containing all coefficients.
	 * @param b The right hand side of the system of equations.
	 */
	public LSqMinNorm(SimpleMatrix a, SimpleVector b){
		assert(a.getRows() == b.getLen());
		this.a = a;
		this.b = b;
		this.nCol = a.getCols();
		this.nRow = a.getRows();
		this.x = new SimpleVector(nCol);
		
		solve();
	}
	
	/**
	 * Solves the system of equations under the constraint of minimal norm of the solution vector.
	 */
	private void solve(){
		// construct new matrix containing the original matrix and then a nCol identity matrix
		SimpleMatrix aPrime = new SimpleMatrix(nRow + nCol, nCol);
		aPrime.setSubMatrixValue(0, 0, a);
		for(int i = 0; i < nCol; i++){
			aPrime.setElementValue(nRow + i, i, Math.sqrt(lambda));
		}
		// construct new right hand side using original right hand side and nCol entries being 0
		SimpleVector bPrime = new SimpleVector(nRow + nCol);
		bPrime.setSubVecValue(0, b);
		
		DecompositionSVD svd = new DecompositionSVD(aPrime);
		SimpleMatrix aPrInv = SimpleOperators.multiplyMatrixProd(svd.getV(), SimpleOperators.multiplyMatrixProd(svd.getreciprocalS(), svd.getU().transposed()));
		
		SimpleVector sol = SimpleOperators.multiply(aPrInv, bPrime);
		
		this.x = sol;
	}
	
	/**
	 * Calculates the root mean square error of the fit using the L2 norm.
	 * @return The error.
	 */
	public double getRmsError(){
		
		SimpleVector opt = new SimpleVector(b.getLen());
		for(int i = 0; i < nCol; i++){
			opt.add(a.getCol(i).multipliedBy(x.getElement(i)));
		}
		opt.subtract(b);
		
		return opt.normL2() / opt.getLen();
	}
	
	/**
	 * Getter for the coefficients solving the system of equations.
	 * @return The coefficients solving the system.
	 */
	public double[] getSolution(){
		return this.x.copyAsDoubleArray();
	}
	
	/**
	 * Setter for the Lagrange multiplier used for regularization.
	 * @param lambda The multiplier.
	 */
	public void setLambda(double lambda){
		this.lambda = lambda;
	}
	
	/**
	 * Getter for the Lagrange Multiplier used for regularization.
	 * @return The multiplier.
	 */
	public double getLambda(){
		return this.lambda;
	}
	
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
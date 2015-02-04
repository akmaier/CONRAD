package edu.stanford.rsl.conrad.calibration;

import java.util.ArrayList;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/**
 * Implements the Principal Component Analysis for a set of PointND.java, restricted to 2D. First the centroid S is computed. 
 * Then the set of points is weighted and the covariance matrix C is built. Eigenvalues and eigenvectors are estimated using
 * the Eigenvalue Decomposition. Imports Jama.Matrix and Jama.EigenvalueDecomposition.
 * 
 * @author Philipp Roser
 */
public class PrincipalComponentAnalysis2D {
	
	/**
	 * double array containing the eigenvalues of the covariance matrix C. Should be of length 2.
	 */
	private double[] eigenvalues;
	
	/**
	 * Jama.Matrix array containing the eigenvectors of the covariance matrix C. Should be of length 2.
	 */
	private Matrix[] eigenvectors;
	
	/**
	 * Covariance matrix C of the data set.
	 */
	private Matrix C;
	
	/**
	 * Centroid S of the data set.
	 */
	private PointND S;
	
	/**
	 * PointND array containing the data to be analyzed.
	 */
	private PointND[] set;
	
	/**
	 * PointND array containing the data translated by the centroid.
	 */
	private PointND[] weightedSet;

	/**
	 * Constructor, expecting the data set to be analyzed as array PointND[]. Executes the complete Principal Component Analysis.
	 * 
	 * @param set
	 */
	public PrincipalComponentAnalysis2D(PointND[] set) {
		this.set = set;
		computeS();
		weightSet();
		buildC();
		computeEigenvalues();
		computeEigenvectors();
	}
	
	
	/**
	 * Constructor, expecting the data set to be analyzed as ArrayList<PointND>. Executes the complete Principal Component Analysis.
	 * 
	 * @param set
	 */
	public PrincipalComponentAnalysis2D(ArrayList<PointND> set) {
		this.set = new PointND[set.size()];
		for (int i = 0; i < this.set.length; i++) {
			this.set[i] = new PointND(set.get(i));
		}
		computeS();
		weightSet();
		buildC();
		computeEigenvalues();
		computeEigenvectors();
	}

	/**
	 * 
	 * @return Principal Components of the data set as Jama.Matrix[] of length 2.
	 */
	public Matrix[] getEigenvectors() {
		return this.eigenvectors;
	}

	/**
	 * 
	 * @return Principal Components of the data set as String.
	 */
	public String stringEigenvectors() {
		String result = "";
		for (int i = 0; i < eigenvectors.length; i++) {
			result += "e" + i + " = (" + eigenvectors[i].get(0, 0) + ", "
					+ eigenvectors[i].get(1, 0) + ")\n";
		}
		return result;
	}

	/**
	 * 
	 * @return eigenvalues of the covariance matrix
	 */
	public double[] getEigenvalues() {
		return this.eigenvalues;
	}

	/**
	 * Computes the centroid S of the data set.
	 */
	private void computeS() {

		double xsum = 0.0;
		double ysum = 0.0;

		for (int i = 0; i < set.length; i++) {
			xsum = xsum + set[i].get(0);
			ysum = ysum + set[i].get(1);
		}

		S = new PointND(xsum / set.length, ysum / set.length);
	}

	/**
	 * Translates all points in the data set by the centroid S.
	 */
	private void weightSet() {
		weightedSet = new PointND[set.length];
		for (int i = 0; i < set.length; i++) {
			weightedSet[i] = new PointND(set[i].get(0) - S.get(0),
					set[i].get(1) - S.get(1));
		}
	}

	/**
	 * Builds the covariance matrix.
	 */
	private void buildC() {
		C = new Matrix(2, 2);
		int n = set.length;
		for (int i = 0; i < n; i++) {
			Matrix P = new Matrix(2, 2);
			P.set(0, 0, ((double) 1.0 / (n - 1)) * weightedSet[i].get(0)
					* weightedSet[i].get(0));
			P.set(0, 1, ((double) 1.0 / (n - 1)) * weightedSet[i].get(0)
					* weightedSet[i].get(1));
			P.set(1, 0, ((double) 1.0 / (n - 1)) * weightedSet[i].get(1)
					* weightedSet[i].get(0));
			P.set(1, 1, ((double) 1.0 / (n - 1)) * weightedSet[i].get(1)
					* weightedSet[i].get(1));
			C = C.plus(P);
		}

	}

	/**
	 * Estimates the eigenvalues of the covariance Matrix C using Jama.EigenvalueDecomposition.
	 */
	private void computeEigenvalues() {
		EigenvalueDecomposition eig = C.eig();
		eigenvalues = eig.getRealEigenvalues();
	}

	/**
	 * Estimates the eigenvectors of the covaricance matrix C using Jama.EigenvalueDecomposition.
	 */
	private void computeEigenvectors() {
		// Matrix c1 = C.minus(Matrix.identity(2, 2).times(eigenvalues[0]));
		// Matrix c2 = C.minus(Matrix.identity(2, 2).times(eigenvalues[1]));
		EigenvalueDecomposition m = C.eig();
		Matrix eig = m.getV();
		eigenvectors = new Matrix[2];
		eigenvectors[0] = eig.getMatrix(0, 1, 0, 0);
		eigenvectors[1] = eig.getMatrix(0, 1, 1, 1);
		/*
		 * eigenvectors[0] = new Matrix(2, 1); eigenvectors[0].set(0, 0,
		 * 1.0); eigenvectors[0].set(1, 0, -1.0 * c1.get(0, 0) / c1.get(0,
		 * 1)); eigenvectors[1] = new Matrix(2, 1); eigenvectors[1].set(0,
		 * 0, 1.0); eigenvectors[1].set(1, 0, -1.0 * c2.get(0, 0) /
		 * c2.get(0, 1));
		 */
	}

}

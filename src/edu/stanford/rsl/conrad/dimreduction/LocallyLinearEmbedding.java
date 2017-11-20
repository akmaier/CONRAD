package edu.stanford.rsl.conrad.dimreduction;


import java.util.ArrayList;

//nach Nonlinear
// Dimensionality
// Reduction by
// Locally Linear
// Embedding von Sam
// T. Roweis and
// Lawrence K. Saul

//author: Susanne Westphal

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.MyException;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class LocallyLinearEmbedding {

	private double[][] distanceMap;
	private int[][] neighbor;
	private PointND[] points; 
	private int numNeighbors;
	private ArrayList<PointND> Y = new ArrayList<PointND>();

	
	/**
	 * Locally Linear Embedding
	 * 
	 * @param distanceMap
	 *            of the original points
	 * @param points
	 *            coordinates of the original points
	 */
	public LocallyLinearEmbedding(PointND[] points) {
		this.points = points;
		this.distanceMap = HelperClass.buildDistanceMatrix(points); 
	}
	/**
	 * sets the number of neighbors of each point
	 * @param numNeighbors
	 */
	public void setNumNeighbors(int numNeighbors){
		this.numNeighbors = numNeighbors; 
		
	}
	/**
	 * 
	 * @return the number of neighbors of each point
	 */
	public int getNumNeighbors(){
		return this. numNeighbors; 
	}
	/**
	 * starts the computation
	 * @throws MyException
	 */
	public void computeLLE() throws MyException{
		neighbor = new int[distanceMap.length][numNeighbors];
		LLE();
	}
	
	/**
	 * main function
	 * 
	 * @throws MyException
	 */
	private void LLE() throws MyException {
		Matrix w = new Matrix(this.points.length, this.numNeighbors);
		Matrix wSquare = null;
		for (int i = 0; i < points.length; ++i) {
			PointND[] list = computenearestNeighbors(this.numNeighbors, i,
					points);
			Matrix gram = computeGramMatrix(list, points[i]);
			Matrix inverse = computeInverse(gram);

			double[] ones = new double[inverse.getColumnDimension()];
			for (int j = 0; j < inverse.getColumnDimension(); ++j) {
				ones[j] = 1.0;
			}
			PointND oneVector = new PointND(ones);
			PointND weights = MatrixVector(inverse, oneVector);
			for (int j = 0; j < inverse.getColumnDimension(); ++j) {
				w.set(i, j, weights.get(j));
			}

			double sum = 0.0;
			for (int j = 0; j < w.getColumnDimension(); ++j) {
				sum += w.get(i, j);
			}
			for (int j = 0; j < w.getColumnDimension(); ++j) {
				w.set(i, j, w.get(i, j) / sum);
			}
		}
		wSquare = fillUp(w);
		Matrix m = computeM(wSquare);
		computeY(m);

	}
	

	/**
	 * finds the n nearest neighbors of point number i of points
	 * 
	 * @param n
	 *            number of neighbors
	 * @param i
	 *            index of the actual point
	 * @param points
	 *            all points
	 * @return PointND[] with the nearest neighbors
	 * @throws MyException
	 */
	private PointND[] computenearestNeighbors(int n, int i, PointND[] points)
			throws MyException {
		if (n > distanceMap[0].length) {
			throw new MyException(
					"LLE: number of neighbors is higher than number of points!!");
		} else {

			ArrayList<Integer> neighbors = new ArrayList<Integer>();
			neighbors.add(i);
			PointND[] listOfNeighbors = new PointND[n];
			double min = -1.0;
			int neighb = 0;

			for (int j = 0; j < n; ++j) {
				min = -1;
				for (int k = 0; k < distanceMap[0].length; ++k) {
					if (!neighbors.contains(k)) {
						if (distanceMap[i][k] < min || min == -1.0) {
							min = distanceMap[i][k];
							neighb = k;
						}
					}
				}
				this.neighbor[i][j] = neighb;
				neighbors.add(neighb);
				listOfNeighbors[j] = points[neighb];
			}
			return listOfNeighbors;
		}
	}

	/**
	 * computes the gram matrix of the differences of one point with its
	 * neighbors
	 * 
	 * @param points
	 *            all neighbor points
	 * @param point
	 *            the actual point
	 * @return the gram matrix
	 */
	private Matrix computeGramMatrix(PointND[] points, PointND point) {
		Matrix gramMatrix;
		gramMatrix = new Matrix(points.length, points.length);
		PointND diffi = new PointND(point);
		PointND diffj = new PointND(point);

		for (int i = 0; i < points.length; ++i) {
			for (int j = i; j < points.length; ++j) {
				for (int k = 0; k < point.getDimension(); ++k) {
					diffi.set(k, points[i].get(k) - point.get(k));
					diffj.set(k, points[j].get(k) - point.get(k));
				}

				gramMatrix.set(i, j, (diffi).innerProduct(diffj));
				gramMatrix.set(j, i, (diffi).innerProduct(diffj));

			}
		}
		return gramMatrix;
	}

	/**
	 * Computes the inverse of a matrix using the SVD
	 * 
	 * @param matrix
	 * @return the inverse of matrix
	 */
	private Matrix computeInverse(Matrix matrix) {
		SingularValueDecomposition SVD = matrix.svd();
		Matrix s = SVD.getS();

		double kond = s.get(0, 0)
				/ s.get(s.getRowDimension() - 1, s.getColumnDimension() - 1);
		if (kond > 10000000) {
			for (int i = 0; i < s.getRowDimension(); ++i) {
				matrix.set(i, i, matrix.get(i, i) + 0.001); // if it is no full
															// rank matrix
			}
		}
		return matrix.inverse();
	}

	/**
	 * Multiplication of matrix with point
	 * 
	 * @param matrix
	 * @param point
	 * @return the vector solution = matrix * point
	 */
	private PointND MatrixVector(Matrix matrix, PointND point){
	
		double sum = 0.0;
		double[] sol = new double[matrix.getRowDimension()];
		for (int i = 0; i < matrix.getRowDimension(); ++i) {
			sum = 0.0;
			for (int j = 0; j < matrix.getColumnDimension(); ++j) {
				sum += matrix.get(i, j) * point.get(j);
			}
			sol[i] = sum;
		}
		PointND solution = new PointND(sol);
		return solution;

	}

	

	/**
	 * creates the full W matrix out of w, adds zeros if the points are no
	 * neighbors
	 * 
	 * @param w
	 *            weights of the points if they are neighbors
	 * @return the Matrix W with zeros
	 */
	private Matrix fillUp(Matrix w) {
		Matrix wSquare = new Matrix(distanceMap.length, distanceMap.length);
		for (int i = 0; i < distanceMap.length; ++i) {
			for (int j = 0; j < this.numNeighbors; ++j) {
				wSquare.set(i, neighbor[i][j], w.get(i, j));
			}
		}
		return wSquare;
	}
	
	/**
	 * computes the matrix M out of w : M = (\vec{I} - \vec{W})^T(\vec{I} - \vec{W})
	 * @param w
	 * @return the matrix M
	 */
	private Matrix computeM(Matrix w) {
		Matrix m = new Matrix(w.getRowDimension(), w.getColumnDimension());
		Matrix matrixprod = new Matrix(w.getRowDimension(), w.getRowDimension());
		double sum = 0.0;
		for (int i = 0; i < m.getRowDimension(); ++i) {
			for (int j = 0; j < m.getRowDimension(); ++j) {
				sum = 0.0;
				if (i == j) {
					w.set(i, j, 1);
				} else {
					w.set(i, j, -w.get(i, j));
				}

			}
		}

		for (int i = 0; i < m.getRowDimension(); ++i) {
			for (int j = 0; j < m.getRowDimension(); ++j) {
				sum = 0.0;
				for (int k = 0; k < m.getColumnDimension(); ++k) {
					sum += w.get(k, i) * w.get(k, j);
				}
				matrixprod.set(i, j, sum);
			}
		}

		return matrixprod;
	}
	
	/**
	 * computes Y, which are the Eigenvectors from M
	 * @param M
	 * @return the coordintas of the low dimensional space stored in matrix Y
	 */
	private PointND[] computeY(Matrix M) {
		SingularValueDecomposition SVD = new SingularValueDecomposition(M);
		Matrix U = SVD.getU();
		double[] zero = new double[U.getColumnDimension()];
		for (int i = 0; i < U.getColumnDimension(); ++i) {
			zero[i] = 0;
		}
		PointND y1 = new PointND(zero);
		PointND y2 = new PointND(zero);
		PointND y3 = new PointND(zero);

		for (int i = 0; i < U.getColumnDimension(); ++i) {
			y1.set(i, U.get(i, U.getRowDimension() - 3));
			y2.set(i, U.get(i, U.getRowDimension() - 2));
			y3.set(i, U.get(i, U.getRowDimension() - 1));
		}
		PointND[] y = new PointND[U.getRowDimension()];
		PointND yi = new PointND(0.0, 0.0, 0.0);
		for (int i = 0; i < U.getRowDimension(); ++i) {
			yi.set(0, y1.get(i));
			yi.set(1, y2.get(i));
			yi.set(2, 0.0);
			y[i] = yi.clone();
			this.Y.add(y[i]);
		}

		return y;

	}

	/**
	 * returns the low dimensional points
	 * @return the low dimensional points
	 */
	public ArrayList<PointND> getPoints() {
		return this.Y;
	}

}

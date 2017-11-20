package edu.stanford.rsl.conrad.dimreduction;


import java.util.ArrayList;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.stanford.rsl.conrad.dimreduction.utils.MyException;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class PCA {

	private PointND[] points;
	PointND[] arrayOfPointsInversePCA;
	private double[] principalComponentsArray;
	private int targetDimension;

	private ArrayList<PointND> pointsDim;
	
	/**
	 * sets the target dimension of the Principal Component Analysis
	 * @param dim target dimension of the PCA
	 */
	public void setTargetDimension(int dim) {
		targetDimension = dim;
	}
	/**
	 * 
	 * @return the target dimension of the PCA
	 */
	public int getTargetDimension() {
		return targetDimension;
	}

	/**
	 * computes a Principal Component Analysis of the points points with the
	 * target dimension dim
	 * 
	 * @param dim
	 *            target dimension
	 * @param points
	 * @return an ArrayList<PointND> with the points in dimension dim
	 * @throws MyException
	 */
	public ArrayList<PointND> computePCA(PointND[] points) throws MyException {
		if (targetDimension > points[0].getDimension()) {
			throw new MyException("Target dimension of the PCA too big!");
		} else {
			// Mean Vector, for centering the points
			PointND meanVector = new PointND(points[0]);
			for (int i = 1; i < points.length; ++i) {
				for (int j = 0; j < points[0].getDimension(); ++j) {
					meanVector.set(j, meanVector.get(j) + points[i].get(j));
				}
			}
			for (int j = 0; j < points[0].getDimension(); ++j) {
				meanVector.set(j, meanVector.get(j) / points.length);
			}

			ArrayList<PointND> principalComponents = new ArrayList<PointND>();
			this.points = new PointND[points.length];
			int pointNumber = 0;
			// centering and writing the points in a matrix
			Matrix vectors = new Matrix(points.length, points[0].getDimension());
			for (int i = 0; i < points.length; ++i) {
				for (int j = 0; j < points[0].getDimension(); ++j) {
					vectors.set(i, j, points[i].get(j) - meanVector.get(j));
				}
			}

			SingularValueDecomposition SVD = new SingularValueDecomposition(
					vectors);
			Matrix s = SVD.getS();

			principalComponentsArray = new double[Math.min(
					s.getColumnDimension(), s.getRowDimension())];
			System.out.println("Principal Components: ");
			for (int i = 0; i < principalComponentsArray.length; ++i) {
				principalComponentsArray[i] = 1.0 / (points.length - 1.0)
						* Math.pow(s.get(i, i), 2);
				System.out.print(principalComponentsArray[i] + ", ");
			}
			System.out.println();
			Matrix v = SVD.getV();

			// take only the dim first columns of V
			Matrix vnew = new Matrix(v.getRowDimension(), targetDimension);
			for (int i = 0; i < v.getRowDimension(); ++i) {
				for (int j = 0; j < targetDimension; ++j) {
					vnew.set(i, j, v.get(i, j));
				}
			}
			// compute the new coordinates by y = x * v
			Matrix Y = vectors.times(vnew);

			// if the dimension of the new points is smaller than 3 additional
			// zero-coordinates are added because the point cloud viewer can
			// show only 3 or more dimensional points
			double[] rankDimPoint;
			PointND RankDimVector = new PointND();
			double[] pointDim = new double[targetDimension];
			pointsDim = new ArrayList<PointND>();
			if (targetDimension == 1) {
				rankDimPoint = new double[3];
				for (int i = 0; i < Y.getRowDimension(); ++i) {
					rankDimPoint[0] = Y.get(i, 0);
					rankDimPoint[1] = 0.0;
					rankDimPoint[2] = 0.0;

					pointDim[0] = Y.get(i, 0);
					pointsDim.add(new PointND(pointDim).clone());
					RankDimVector = new PointND(rankDimPoint);
					principalComponents.add(RankDimVector.clone());
					this.points[pointNumber] = RankDimVector.clone();
					++pointNumber;
				}
			} else if (targetDimension == 2) {
				rankDimPoint = new double[3];
				for (int i = 0; i < Y.getRowDimension(); ++i) {
					rankDimPoint[0] = Y.get(i, 0);
					rankDimPoint[1] = Y.get(i, 1);
					rankDimPoint[2] = 0.0;

					pointDim[0] = Y.get(i, 0);
					pointDim[1] = Y.get(i, 1);
					pointsDim.add(new PointND(pointDim).clone());
					RankDimVector = new PointND(rankDimPoint);
					principalComponents.add(RankDimVector.clone());
					this.points[pointNumber] = RankDimVector.clone();
					++pointNumber;
				}
			} else {
				rankDimPoint = new double[targetDimension];
				for (int i = 0; i < Y.getRowDimension(); ++i) {
					for (int j = 0; j < targetDimension; ++j) {
						rankDimPoint[j] = Y.get(i, j);
					}

					RankDimVector = new PointND(rankDimPoint);
					principalComponents.add(RankDimVector.clone());
					this.points[pointNumber] = RankDimVector.clone();
					pointsDim.add(RankDimVector.clone());
					++pointNumber;
				}
			}
			return principalComponents;
		}
	}

	/**
	 * returns a PointND[] with the coordinates of the result of the PCA
	 * 
	 * @return a PointND[] with the coordinates of the result of the PCA
	 */
	public PointND[] getPoints() {
		return points;
	}

	/**
	 * returns the size of the principal components
	 * 
	 * @return the size of the principal components
	 */
	public double[] getPrincipalComponents() {
		return principalComponentsArray;
	}

	/**
	 * returns a ArayList of coordinates of the low dimensional points
	 * 
	 * @return a ArrayList<PointND> of the coordinates of the PCA without the
	 *         adding of zeros if the target dimension is smaller than three
	 */
	public ArrayList<PointND> getPointsInDim() {
		return pointsDim;
	}

}

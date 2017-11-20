package edu.stanford.rsl.conrad.dimreduction.utils;


import java.util.ArrayList;
import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.data.Grid;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class HelperClass {

	/**
	 * converts a double array in a arrayList of PointND
	 * 
	 * @param x
	 *            : double array with coordinates
	 * @param numpoints
	 *            : number of points in x
	 * @return a ArrayList<PointND> of the coordinates x
	 */
	public static ArrayList<PointND> wrapArrayToPoints(double[] x, int numpoints) {
		ArrayList<PointND> points = new ArrayList<PointND>();
		int dim = x.length / numpoints;
		double[] p;
		if (dim < 3) {
			p = new double[3];
		} else {
			p = new double[dim];
		}

		PointND point = new PointND(p);

		for (int i = 0; i < numpoints; i++) {
			for (int j = 0; j < dim; ++j) {
				if (dim == 1 && j == 0) {
					point.set(1, 0);
					point.set(2, 0);
				} else if (dim == 2 && j == 0) {
					point.set(2, 0);
				}
				point.set(j, x[i * dim + j]);
			}
			points.add(point.clone());
		}
		return points;
	}

	/**
	 * converts a double array of points in a PointND array
	 * 
	 * @param points
	 *            : double array of coordinates
	 * @param numPoints
	 *            : number of points in points
	 * @return a PointND array with the points in points
	 */
	static PointND[] wrapDoubleToPointND(double[] points, int numPoints) {
		ArrayList<PointND> pointsList = wrapArrayToPoints(points, numPoints);
		PointND[] pointsArray = new PointND[numPoints];
		for (int i = 0; i < pointsList.size(); ++i) {
			pointsArray[i] = pointsList.get(i);
		}
		return pointsArray;
	}

	/**
	 * converts a ArrayList<PointND> in a double array
	 * 
	 * @param points
	 *            : ArrayList<PointND> of points
	 * @return a double array with the coordinates of the points in points
	 * @throws MyException
	 */
	static double[] wrapArrayListToDouble(ArrayList<PointND> points) {
		int numPoints = points.size();
		int dim = points.get(0).getDimension();
		double[] pointsArray = new double[dim * numPoints];
		int counter = 0;
		for (int i = 0; i < numPoints; ++i) {
			for (int j = 0; j < dim; ++j) {
				pointsArray[counter] = points.get(i).get(j);
				++counter;
			}
		}
		return pointsArray;

	}

	/**
	 * converts a ArrayList<PointND> in a PointND array
	 * 
	 * @param points
	 *            : ArrayList<PointND> of points
	 * @return a PointND array of the points
	 * @throws MyException
	 */
	public static PointND[] wrapListToArray(ArrayList<PointND> points) {
		return wrapDoubleToPointND(wrapArrayListToDouble(points), points.size());
	}

	/**
	 * computes the inner product of two 2-dimensional points
	 * 
	 * @param x
	 *            array of many coordinates
	 * @param p
	 *            number of the first point of the inner-product
	 * @param q
	 *            number of the second point of the inner-product
	 * @return the inner-product of the two points
	 */
	public static double innerProduct2D(double[] x, int p, int q) {
		return (x[p * 2] * x[q * 2]) + (x[(p * 2) + 1] * x[(q * 2) + 1]);

	}

	/**
	 * computes the euclidean distance of two dim-dimensional points
	 * 
	 * @param x
	 *            array of coordinates
	 * @param p
	 *            number of the first point
	 * @param q
	 *            number of the second point
	 * @param dim
	 *            dimension of the points
	 * @return the euclidean distance of the two points
	 */
	public static double distance(double[] x, int p, int q, int dim) {
		double sum = 0.0;
		for (int i = 0; i < dim; ++i) {
			sum += Math.pow(x[p * dim + i] - x[q * dim + i], 2);
		}
		return Math.sqrt(sum);
	}

	/**
	 * computes the euclidean distance of two 2-dimensional points
	 * 
	 * @param x
	 *            array of coordinates
	 * @param p
	 *            number of the first point
	 * @param q
	 *            number of the second point
	 * @return the euclidean distance of the two points
	 */
	public static double distance2D(double[] x, int p, int q) {
		return Math.sqrt(Math.pow(x[p * 2] - x[q * 2], 2)
				+ Math.pow(x[(p * 2) + 1] - x[(q * 2) + 1], 2));
	}

	/**
	 * shows the points in a PointCloudViewer adds dimensions in case of
	 * dimension < 3
	 * 
	 * @param result
	 *            points that are going to be displayed
	 * @param numpoints
	 *            number of points
	 * @param function
	 *            title of the function
	 * @return the Point Cloud viewer with the points
	 */
	public static PointCloudViewer showPoints(double[] result, int numpoints,
			String function) {
		double[] resultDim;
		if (result.length / numpoints == 2) {
			resultDim = new double[numpoints * 3];
			for (int i = 0; i < numpoints; ++i) {
				resultDim[i * 3] = result[2 * i];
				resultDim[i * 3 + 1] = result[2 * i + 1];
				resultDim[i * 3 + 2] = 0;
			}
		} else if (result.length / numpoints == 1) {
			resultDim = new double[numpoints * 3];
			for (int i = 0; i < numpoints; ++i) {
				resultDim[i * 3] = result[i];
				resultDim[i * 3 + 1] = 0;
				resultDim[i * 3 + 2] = 0;
			}
		} else {
			resultDim = result;
		}
		ArrayList<PointND> points2 = wrapArrayToPoints(resultDim, numpoints);
		PointCloudViewer pcv = new PointCloudViewer(function, points2);
		pcv.setVisible(true);
		return pcv;
	}

	/**
	 * builds the distance matrix based on the coordinates of the points, they
	 * are scaled to a range of [0; 4.0], this leads to a numerical stable
	 * solution
	 * 
	 * @param points
	 * @return a matrix of the inner point distances
	 */
	public static double[][] buildDistanceMatrix(PointND[] points) {
		double[][] distances = new double[points.length][points.length];
		for (int i = 0; i < points.length; i++) {
			for (int j = 0; j < points.length; j++) {
				distances[i][j] = points[i].euclideanDistance(points[j]);
			}
		}
		return distances;
	}
	
	/**
	 * builds the distance matrix based input Grid
	 * 
	 * @param grids
	 * @return a matrix of the inner point distances
	 */
	public static double[][] buildDistanceMatrix(ArrayList<NumericGrid> theGrids) {
		double[][] distances = new double[theGrids.size()][theGrids.size()];
		for (int i = 0; i < theGrids.size(); i++) {
			for (int j = 0; j < theGrids.size(); j++) {
				NumericGrid dist = theGrids.get(i).clone();
				NumericGridOperator op = dist.getGridOperator();
				op.subtractBy(dist, theGrids.get(j));
				distances[i][j] = op.normL2(dist) / dist.getNumberOfElements();
			}
		}
		return distances;
	}
	
	/**
	 * normalizes the inner point distances of the point in the distance matrix, such that die maximum inner-point distance is maxDist
	 * @param distances double[][] of inner-point distances
	 * @param maxDist maximum inner-point distance after normalization
	 * @return the normalized distance matrix
	 */
	static double[][] normalizeInnerPointDistanceMax(double[][] distances, double maxDist){
		
		double max = -1;
		for(int i = 0; i < distances.length; ++i){
			for(int j = i +1; j < distances[i].length; ++j){
				if(max < distances[i][j] || max == -1){
					max = distances[i][j];
				}
			}
		}
		for(int i = 0; i < distances.length; ++i){
			for(int j = 0; j < distances[i].length; ++j){
				distances[i][j] = distances[i][j] / max * maxDist;
			}
		}
		return distances;
		
	}
	
	/**
	 * normalizes the inner point distances of the point in the distance matrix, such that die mean inner-point distance is meanDist
	 * @param distances double[][] of inner-point distances
	 * @param meanDist mean inner-point distance after normalization
	 * @return the normalized distance matrix
	 */
	static double[][] normalizeInnerPointDistanceMean(double[][] distances, double meanDist){
		
		double mean = 0.0; 
		for(int i = 0; i < distances.length; ++i){
			for(int j = i+1; j < distances[i].length; ++j){
				mean += distances[i][j];
			}
		}
		mean /= ((distances.length * distances.length - 1)/2);
		for(int i = 0; i < distances.length; ++i){
			for(int j = 0; j < distances[i].length; ++j){
				distances[i][j] = distances[i][j] / mean * meanDist;
			}
		}
		return distances; 
	}
}
package edu.stanford.rsl.conrad.dimreduction.utils;


import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class PointCloud {

	ArrayList<PointND> points;
	/**
	 * Constructor, if a comma separated list of PointND is given
	 * @param points
	 */
	public PointCloud(PointND... points) {
		this.points = new ArrayList<PointND>();
		for (PointND point : points)
			this.points.add(point);
	}
	/**
	 * Constructor, is an ArrayList<PointND> is given
	 * @param points
	 */
	public PointCloud(ArrayList<PointND> points) {
		this.points = points;
	}
	
	/**
	 * 
	 * @return the ArrayList<PointND> of the PointCloud
	 */
	public ArrayList<PointND> getPoints() {
		return points;
	}
	
	/**
	 * returns the distance matrix of the points in the PointCloud
	 * @return the distace matrix of the PointCloud
	 */
	public double[][] getDistanceMatrix(){
		return HelperClass.buildDistanceMatrix((PointND[])points.toArray()); 
	}
	
	/**
	 * function to center the points around the origin
	 */
	public void centerPoints(){
		double[] a  = new double[points.get(0).getDimension()];
		PointND avg = new PointND(a);
		for(int i = 0; i < points.size(); ++i){
			for(int j = 0; j < points.get(i).getDimension(); ++j){
				avg.set(j, avg.get(j) + points.get(i).get(j));
			}
		}
		for(int i = 0; i < avg.getDimension(); ++i){
			avg.set(i, avg.get(i) / points.size());
		}
		for(int i = 0; i < points.size(); ++i){
			for(int j = 0; j < points.get(i).getDimension(); ++j){
				points.get(i).set(j, points.get(i).get(j) - avg.get(j));
			}
		}
	}
	
	/**
	 * normalizes the mean inner point distance of all points in the cloud to the parameter value
	 * @param meanDistance mean inner point distance of all point in the PointCloud
	 */
	public void normalizeInnerPointDistancesMean(double meanDistance){ 
		this.centerPoints();
		double[][] distances = HelperClass.buildDistanceMatrix(HelperClass.wrapListToArray(points));
		double mean = 0.0; 
		for(int i = 0; i < distances.length; ++i){
			for(int j = i+1; j < distances[i].length; ++j){
				mean += distances[i][j];
			}
		}
		mean /= ((distances.length * distances.length - 1)/2);
		for(int i = 0; i < points.size(); ++i){
			for(int j = 0; j < points.get(i).getDimension(); ++j){
				points.get(i).set(j, points.get(i).get(j)/mean * meanDistance);
			}
		}
	}
	
	/**
	 * normalizes the maximal inner point distance of all point to the parameter value 
	 * @param maxDistance maximal inner point distance to scale the distances
	 */
	public void normalizeInnerPointDistancesMax(double maxDistance){
		this.centerPoints();
		double[][] distances = HelperClass.buildDistanceMatrix(HelperClass.wrapListToArray(points));
		double max = -1;
		for(int i = 0; i < distances.length; ++i){
			for(int j = i +1; j < distances[i].length; ++j){
				if(max < distances[i][j] || max == -1){
					max = distances[i][j];
				}
			}
		}
		for(int i = 0; i < points.size(); ++i){
			for(int j = 0; j < points.get(i).getDimension(); ++j){
				points.get(i).set(j, points.get(i).get(j)/max * maxDistance);
			}
		}
	}

}
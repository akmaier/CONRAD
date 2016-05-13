/*
 * Copyright (C) 2014 Susanne Westphal 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Class to model points of arbitrary dimension. Compatible with numerics.
 * 
 * @author suwestphal
 * 
 */
public class SwissRoll implements Serializable {

	private static final long serialVersionUID = 1L;
	private static PointND[] points;
	private static ArrayList<PointND> pointList;

	/**
	 * function to build a SwissRoll with the given parameters, saves the points in an PointND[] array and in a ArrayList<PoinND>  
	 * @param gap gap between the single "snakes"
	 * @param numberOfPoints number of points in one "snake"
	 * @param width number of "snakes" beside each other
	 */
	public SwissRoll(double gap, int numberOfPoints, int width) {
		pointList = new ArrayList<PointND>();
		points = new PointND[numberOfPoints * width];
		double distance = 1.0 / numberOfPoints;
	
		for (int i = 0; i < numberOfPoints; ++i) {
			for (int j = 0; j < width; ++j) {
				points[width * i + j] = new PointND(5
						* Math.sqrt(2 + 2 * (-1 + i * distance))
						* Math.cos(2 * Math.PI
								* Math.sqrt(2 + 2 * (-1 + i * distance))), 5
						* Math.sqrt(2 + 2 * (-1 + i * distance))
						* Math.sin(2 * Math.PI
								* Math.sqrt(2 + 2 * (-1 + i * distance))), 2
						* j * gap);
				pointList.add(points[width * i + j]);

			}
		}
	}
	/**
	 * returns an ArrayList of the points of the SwissRoll
	 * @return an ArrayList of the points of the SwissRoll
	 */
	public ArrayList<PointND> getPointList(){
		return pointList; 
	}
	
	/**
	 * returns an PointND[] of the points of the SwissRoll
	 * @return an PointND[] of the points of the SwissRoll
	 */
	public PointND[] getPoints(){
		return points;
	}

}
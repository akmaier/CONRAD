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

	public ArrayList<PointND> buildSwissRoll(double gap, int numberOfPoints, int width) {
		ArrayList<PointND> Swiss = new ArrayList<PointND>();
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
				Swiss.add(points[width * i + j]);

			}
		}
		return Swiss;

	}

	public PointND[] getPoints(){
		return points;
	}

}
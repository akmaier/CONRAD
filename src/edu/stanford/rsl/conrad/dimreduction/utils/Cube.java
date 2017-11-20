/*
 * Copyright (C) 2014 Susanne Westphal 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.dimreduction.utils;


import java.util.ArrayList;
import java.util.Random;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class Cube {

	private static PointND[] points;
	private static ArrayList<PointND> pointsList = new ArrayList<PointND>();

	/**
	 * Builds an Cube with chosen parameters, width, dimension, standard
	 * deviation, number of points per corner, saved points (if desired), save
	 * the random points (if desired), add additional dimensions of noise.
	 * 
	 * @param width
	 *            edge length of the cube
	 * @param dim
	 *            dimension of the cube
	 * @param standarddeviation
	 *            standard deviation of the point clouds
	 * @param numberOfCloudPoints
	 *            number of point per corner/cloud
	 * @param computeWithSavedPoints
	 *            takes saved points with the same parameters, if already saved,
	 *            will throw an error otherwise
	 * @param savePoints
	 *            saves the points with the given parameters
	 * @param noiseDim
	 *            additional dimension for the noise of the points
	 */
	public Cube(double width, int dim, double standarddeviation, int numberOfCloudPoints,
			boolean computeWithSavedPoints, boolean savePoints, int noiseDim) {

		points = new PointND[(int) Math.pow(2, dim) * numberOfCloudPoints];

		String filename = "";
		if (savePoints || computeWithSavedPoints) {
			filename = Double.toString(width);
			filename += Integer.toString(dim);
			filename += Double.toString(standarddeviation);
			filename += Integer.toString(numberOfCloudPoints);
		}

		if (!computeWithSavedPoints) {

			PointND point = new PointND();

			if (dim + noiseDim < 3) {
				double[] x = new double[3];
				x[2] = 0;
				x[1] = 0;
				for (int j = 0; j < Math.pow(2, dim); ++j) {
					String binaer = Integer.toBinaryString(j);
					int length = binaer.length();
					for (int l = length; l < dim; ++l) {
						binaer = "0" + binaer;
					}

					for (int i = 0; i < numberOfCloudPoints; ++i) {

						Random random = new Random();
						for (int m = 0; m < dim; ++m) {
							if (binaer.charAt(m) == '1') {
								x[m] = random.nextGaussian() * standarddeviation + width / 2.0;
							} else {
								x[m] = random.nextGaussian() * standarddeviation - width / 2.0;
							}
						}
						for (int m = dim; m < dim + noiseDim; ++m) {
							x[m] = random.nextGaussian() * standarddeviation;
						}

						point = new PointND(x);
						points[j * numberOfCloudPoints + i] = point.clone();
						pointsList.add(points[j * numberOfCloudPoints + i]);

					}
				}

			} else {
				double[] x = new double[dim + noiseDim];
				for (int j = 0; j < Math.pow(2, dim); ++j) {
					String binaer = Integer.toBinaryString(j);
					int length = binaer.length();
					for (int l = length; l < dim; ++l) {
						binaer = "0" + binaer;
					}

					for (int i = 0; i < numberOfCloudPoints; ++i) {

						Random random = new Random();
						for (int m = 0; m < dim; ++m) {
							if (binaer.charAt(m) == '1') {
								x[m] = random.nextGaussian() * standarddeviation + width / 2.0;
							} else {
								x[m] = random.nextGaussian() * standarddeviation - width / 2.0;
							}
						}

						for (int m = dim; m < (noiseDim + dim); ++m) {
							x[m] = random.nextGaussian() * standarddeviation;
						}

						point = new PointND(x);
						points[j * numberOfCloudPoints + i] = point.clone();
						pointsList.add(points[j * numberOfCloudPoints + i]);

					}
				}
			}
			if (savePoints) {
				FileHandler.save(filename, points);
			}

		} else {
			points = FileHandler.readPointND(filename);
			pointsList = new ArrayList<PointND>();

			try {
				for (int i = 0; i < points.length; ++i) {
					pointsList.add(points[i]);
				}
			} catch (NullPointerException e) {
				System.err.println("Cube null " + e.getMessage());
			}

		}
	}

	/**
	 * Returns an ArrayList<PointND> of all points of the Cube
	 * 
	 * @return ArrayList<PointND> of all points of the Cube
	 */
	public ArrayList<PointND> getPointList() {
		return pointsList;
	}

	/**
	 * Returns an array of PointND of all points of the Cube
	 * 
	 * @return an PointND[] of all points of the Cube
	 */
	public PointND[] getPoints() {
		return points;
	}

}
package edu.stanford.rsl.conrad.dimreduction.utils;


import java.io.IOException;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class ScatterPlot {
	private double[] x;
	private double[] y;
	private double[] z;
	private int length;
	int index = 0;

	/**
	 * Constructor
	 * 
	 * @param length
	 *            number of points that will be saved in the coordinates x, y, z
	 */
	public ScatterPlot(int length) {
		this.length = length;
		x = new double[length];
		y = new double[length];
		z = new double[length];

	}

	/**
	 * function to find the minimum number in an array
	 * 
	 * @param array
	 * @return the index of the minimum number in array
	 */
	public int findMinimum(double[] array) {
		int indexMin = 0;
		double min = array[0];
		for (int i = 1; i < array.length; ++i) {
			if (array[i] < min) {
				min = array[i];
				indexMin = i;
			}
		}
		return indexMin;
	}

	/**
	 * Adds a single point to the array
	 * 
	 * @param x
	 *            x-coordinate
	 * @param y
	 *            y-coordinate
	 * @param z
	 *            z-coordinate
	 * @param name
	 *            filename if it will be saved
	 * @throws IOException
	 */
	public void addValue(double x, double y, double z, String name) throws IOException {
		this.x[index] = x;
		this.y[index] = y;
		this.z[index] = z;
		++index;
		if (index == length && name.length() != 0) {
			FileHandler.save(this.x, this.y, this.z, name);
		}

	}

	/**
	 * adds points to the array
	 * 
	 * @param x
	 *            x-coordinates of the points
	 * @param y
	 *            y-coordinates of the points
	 * @param z
	 *            z-coordinates of the points
	 * @param name
	 *            filename if it will be saved
	 * @throws IOException
	 */
	public void addValue(double[] x, double[] y, double[] z, String name) throws IOException {
		for (int i = 0; i < x.length; ++i) {
			this.x[index] = x[i];
			this.y[index] = y[i];
			this.z[index] = z[i];
			++index;
		}
		if (index == length && name.length() != 0) {

			FileHandler.save(this.x, this.y, this.z, name);
		}
	}

}
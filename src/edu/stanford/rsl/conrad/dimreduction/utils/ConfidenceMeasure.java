package edu.stanford.rsl.conrad.dimreduction.utils;


import java.io.IOException;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class ConfidenceMeasure {

	/**
	 * Starts the computation of the confidence measure. Which is a measurement
	 * of the difference in the distance matrices of the original and the low
	 * dimensional points.
	 * 
	 * @param original
	 *            : distance Matrix of the high-dimensional points
	 * @param reconstructed
	 *            : distance matrix of the low-dimensional points
	 * @throws IOException
	 */
	public ConfidenceMeasure(double[][] original, double[][] reconstructed) throws IOException {

		double[][] differenceMatrix = differenceMatrix(original, reconstructed);
		saveDifferenceMatrix(differenceMatrix);
	}

	/**
	 * Function to compute the percental difference in the distances of the
	 * high and low dimensional points
	 * 
	 * @param original
	 *            : distance matrix of the high-dimensional points
	 * @param reconstructed
	 *            : distance matrix of the low-dimensional points
	 * @return the percentage difference of the matrices
	 */
	private double[][] differenceMatrix(double[][] original, double[][] reconstructed) {
		double[][] differenceMatrix = new double[original.length][original[0].length];
		for (int i = 0; i < original.length; ++i) {
			for (int j = 0; j < original[0].length; ++j) {
				differenceMatrix[i][j] = Math
						.abs((original[i][j] - reconstructed[i][j]) / (original[i][j] + 0.0000001));
			}
		}
		return differenceMatrix;
	}

	/**
	 * saves the difference matrix in x,y,z notation. x is the row, y the
	 * column, and z the value
	 * 
	 * @param differences
	 *            : matrix that is saved
	 * @throws IOException
	 */
	private void saveDifferenceMatrix(double[][] differences) throws IOException {
		double[] x = new double[differences.length * differences[0].length];
		double[] y = new double[differences.length * differences[0].length];
		double[] z = new double[differences.length * differences[0].length];

		for (int i = 0; i < differences.length; ++i) {
			for (int j = 0; j < differences[0].length; ++j) {
				x[i * differences.length + j] = i;
				y[i * differences.length + j] = j;
				z[i * differences.length + j] = differences[i][j];
			}
		}
		FileHandler.save(x, y, z, "Confidence");
	}

}
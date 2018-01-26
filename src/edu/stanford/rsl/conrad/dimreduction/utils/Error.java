package edu.stanford.rsl.conrad.dimreduction.utils;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class Error {

	public Error() {
	}

	/**
	 * Computes the Sammon Error of the computed points based on the original
	 * distances
	 * 
	 * @param computedPoints
	 *            coordinates of the computed points
	 * @param title
	 *             title of the window
	 * @param distances
	 *             original distances in the high-dimensional space
	 * @return the Sammon Error
	 */
	public double computeError(ArrayList<PointND> computedPoints, String title,
			double[][] distances) {
		// Maximum of the orignial Distances
		// double maxDistanceOriginalPoints = 0.0;
		double sumDistancesOriginalPoints = 0.0;
		for (int i = 0; i < distances.length; ++i) {
			for (int j = i + 1; j < distances.length; ++j) {
				// if (distances[i][j] > maxDistanceOriginalPoints) {
				// maxDistanceOriginalPoints = distances[i][j];
				// }
				if (distances[i][j] == 0) {
					distances[i][j] = 0.0000001;
				}
				sumDistancesOriginalPoints += distances[i][j];
			}
		}

		double averageDistanceOriginalPoints = sumDistancesOriginalPoints
				/ distances.length;

		// Distances of the reconstructed Points
		double[][] distancesResult = new double[computedPoints.size()][computedPoints
				.size()];
		for (int i = 0; i < computedPoints.size(); i++) {
			for (int j = i + 1; j < computedPoints.size(); j++) {
				distancesResult[i][j] = computedPoints.get(i)
						.euclideanDistance(computedPoints.get(j));
			}
		}

		// Maximum of the Distances of the reconstructed points:
		// double maxDistanceReconstructedPoints = 0.0;
		double sumDistancesReconstructedPoints = 0.0;
		for (int i = 0; i < distancesResult.length; ++i) {
			for (int j = i + 1; j < distancesResult.length; ++j) {
				// if (distancesResult[i][j] > maxDistanceReconstructedPoints) {
				// maxDistanceReconstructedPoints = distancesResult[i][j];
				// }
				sumDistancesReconstructedPoints += distancesResult[i][j];
			}
		}
		double averageDistancesReconstructedPoints = sumDistancesReconstructedPoints
				/ distances.length;
		// scalingFactor from reconstructed points to original points
		double scalingFactor = averageDistanceOriginalPoints
				/ averageDistancesReconstructedPoints;

		// System.out.println(scalingFactor);
		// or scalingFactor =
		// maxDistanceOriginalPoints/maxDistanceReconstructedPoints
		// but the results are worse with this scaling

		// Sammon Error function with the scaled distances
		double sum = 0.0;
		double sumDistances = 0.0;
		for (int i = 0; i < distancesResult.length; ++i) {
			for (int j = i + 1; j < distancesResult[0].length; ++j) {
				sum += ((distancesResult[i][j] * scalingFactor - distances[i][j]) * (distancesResult[i][j]
						* scalingFactor - distances[i][j]))
						/ distances[i][j];
				sumDistances += distances[i][j];
			}
		}

		return sum / sumDistances;
	}

}
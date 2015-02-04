/*
 * Copyright (C) 2014 Madgalena Herbst
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.noncircularfov;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;

/**
 * Implementation of an algorithm that completes a given sinogram by using
 * alpha2 = -alpha1 and beta2 = beta1 + pi + 2 alpha1. One method completes the
 * sinogram and another method shows the double acquired data. Another method
 * finds the minimal complete set for a given sinogram and detector size.
 * 
 * @author Magdalena Herbst
 * 
 */

public class Correspondences {

	// parameters of the scanning geomtry
	private double focalLength;
	private double deltaBeta, deltaT;

	int maxTIndex, maxBetaIndex;

	/**
	 * Focal length is defined in the constructor.
	 * 
	 * @param focalLength
	 */

	public Correspondences(double focalLength, Grid2D sino) {
		this.focalLength = focalLength;

		// set other parameters according to the sinogram.
		maxTIndex = sino.getWidth();
		maxBetaIndex = sino.getHeight();
		deltaT = sino.getSpacing()[0];
		deltaBeta = sino.getSpacing()[1];
	}

	/**
	 * Method that displays the double acquired data. Double acquired data is
	 * colored gray, missing areas or the result of zero-line-integrals are
	 * colored black. The given sinogram has to be over a rotation of 360
	 * degrees (it is the ground truth). The mask is a binary mask that is 1 at
	 * the positions where data was acquired. The mask and the sinogram must
	 * have the same dimensions.
	 * 
	 * @param mask
	 * @param sino
	 * @return the redundant areas
	 * @throws IllegalArgumentException
	 */
	public Grid2D findRedundancies(Grid2D mask, Grid2D sino)
			throws IllegalArgumentException {

		if (mask.getHeight() != sino.getHeight()
				|| mask.getWidth() != sino.getWidth()) {
			throw new IllegalArgumentException("Dimensions must agree!");
		}

		double mid = maxTIndex / 2.0;
		Grid2D erg = new Grid2D(maxTIndex, maxBetaIndex);
		erg.setSpacing(deltaT, deltaBeta);

		for (int y = 0; y < maxBetaIndex; y++) {
			for (int x = 0; x < maxTIndex; x++) {

				if (mask.getAtIndex(x, y) > 0 && sino.getAtIndex(x, y) > 0) {
					int xc = (int) (maxTIndex - 1) - x;
					double alpha = (Math.atan2((x - mid) * deltaT, focalLength));
					double betac = y * deltaBeta + Math.PI + 2 * alpha;
					betac = betac % (2 * Math.PI);
					if (erg.getAtIndex(x, y) > 0) {
						erg.setAtIndex(x, y, .5f);
						erg.setAtIndex(xc, (int) (betac / deltaBeta), .5f);
					} else {
						erg.setAtIndex(x, y, 1.f);
						erg.setAtIndex(xc, (int) (betac / deltaBeta), 1.f);
					}

				}
			}
		}
		return erg;
	}

	/**
	 * Methods that completes a given sinogram. The given sinogram has to be
	 * over a rotation of 360 degrees (it is the ground truth). The mask is a
	 * binary mask that is 1 at the positions where data was acquired. The mask
	 * and the sinogram must have the same dimensions.
	 * 
	 * @param mask
	 * @param sino
	 * @return the completed sinogram
	 * @throws IllegalArgumentException
	 */
	public Grid2D findCorrespondences(Grid2D mask, Grid2D sino)
			throws IllegalArgumentException {

		if ((mask.getHeight() != sino.getHeight())
				|| (mask.getWidth() != sino.getWidth())) {
			throw new IllegalArgumentException("Dimensions must agree!");
		}

		maxTIndex = sino.getWidth();
		maxBetaIndex = sino.getHeight();
		deltaT = sino.getSpacing()[0];
		deltaBeta = sino.getSpacing()[1];

		double mid = maxTIndex / 2.0;
		Grid2D erg = new Grid2D(maxTIndex, maxBetaIndex);
		erg.setSpacing(deltaT, deltaBeta);

		// first set acquired values
		for (int y = 0; y < maxBetaIndex; y++) {
			for (int x = 0; x < maxTIndex; x++) {
				if (mask.getAtIndex(x, y) > 0) {
					erg.setAtIndex(x, y, sino.getAtIndex(x, y));
				}
			}
		}

		// then compute the missing values using the corresponding values
		for (int y = 0; y < maxBetaIndex; y++) {
			for (int x = 0; x < maxTIndex; x++) {
				if (mask.getAtIndex(x, y) == 0) {
					double xc = (maxTIndex - 1) - x;
					double alpha = (Math.atan2((x - mid) * deltaT, focalLength));
					double yc = y * deltaBeta + Math.PI + 2 * alpha;
					yc = yc % (2 * Math.PI);

					float val = InterpolationOperators.interpolateLinear(erg,
							xc, yc / deltaBeta);
					erg.setAtIndex(x, y, val);
				}
			}
		}

		return erg;
	}

	/**
	 * Method that computes the missing pixels that result if segments of a
	 * given sinogram are completed with findCorrespondences(...). The detector
	 * follows the left boundary.
	 * 
	 * @param sino
	 *            ground truth sinogram, has to be over a complete rotation.
	 * @param start
	 *            index of starting point
	 * @param rotationLength
	 *            in pixels
	 * @param size
	 *            the detector size in pixels
	 * @return number of missing pixels
	 */
	public int missing(Grid2D sino, int start, int rotationLength, int size) {

		int width = sino.getWidth();
		int height = sino.getHeight();

		// Create mask. Detector follows the left boundary.
		Grid2D mask = new Grid2D(width, height);
		for (int y = start; y < start + rotationLength; y++) {
			for (int x = 0; x < width; x++) {

				// y % height to simulate a never-ending sinogram
				//if (sino.getAtIndex(x, y % height) > 0) {
				if ((x>150)&&(x<150+size)) {
					for (int i = 0; i < size; i++) {
						if (x + i < width) {
							mask.setAtIndex(x + i, y % height, 1.f);
						}
					}

					break;
				}

			}
		}

		// Complete the sinogram.
		Grid2D res = findCorrespondences(mask, sino);


		int missing = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {

				// count the pixels that are zero in the completed image but not
				// in the ground truth image
				if (sino.getAtIndex(x, y) > 0 && res.getAtIndex(x, y) == 0) {
					missing++;
				}
			}
		}

		return missing;
	}

	/**
	 * Method that performs a grid search to find the minimal complete dataset.
	 * A better way to choose the minimal set may be found, the current one is
	 * not perfect. Note that this method may have a runtime up to 15 minutes
	 * because it is not optimized.
	 * 
	 * @param sino
	 * @param size
	 *            size of the detector in pixel
	 * @return result[0] = required rotation, result[1] = corresp. starting
	 *         point, result[2] = corresp. missing pixel
	 */
	public int[] evaluateNumer(Grid2D sino, int size) {

		int a = sino.getHeight() / 2; // start at 180 degrees
		int b = sino.getHeight();

		int[] werteStart = new int[b - a];
		int[] werteMiss = new int[b - a];
		int[] res = new int[3];

		for (int rot = a; rot < b; rot++) {

			// find best starting point for current rotationLength
			int minMiss = Integer.MAX_VALUE;
			int minMissStart = 0;

			for (int start = 0; start < 360; start++) {
				int miss = missing(sino, start, rot, size);

				if (miss < minMiss) {
					minMiss = miss;
					minMissStart = start;
				}
			}

			werteStart[rot - a] = minMissStart;
			werteMiss[rot - a] = minMiss;

		}

		int min_idx = 0;

		// uncomment the output to see the missing pixel for the other
		// settings. The chosen minimal set might not be the best.

		// System.out.println("rot\t" + "start\t" + "missing");
		for (int i = 0; i < werteStart.length; i++) {
			// System.out.println((i + a) + "\t" + werteStart[i] + "\t" +
			// werteMiss[i]);

			if (i != 0) {
				// choose the setting where the number of missing pixel does not
				// change any more (or only a little)
				int change = 4; // or 0, 1, 2, 3,... did not find the perfect
				// solution yet.
				if (werteMiss[i] == werteMiss[i - 1] - change) {
					min_idx = i;
				}
			}
		}

		res[0] = min_idx + a;
		res[1] = werteStart[min_idx];
		res[2] = werteMiss[min_idx];
		return res;
	}

}

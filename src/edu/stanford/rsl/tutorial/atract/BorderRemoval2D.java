package edu.stanford.rsl.tutorial.atract;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

/**
 * This class implements a method to remove the borders caused by the Collimator. 
 * Use this for 2-D detectors.
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class BorderRemoval2D {
	/**
	 * Use this method in combination with the Laplace2D Filter.
	 * This method searches for the collimator borders in a laplacian filtered image.
	 * The current implementation searches in each detector for the first and last non-zero element.
	 * The coordinates with the maximum number of non-zero elements across all projections are then assumed to be the 
	 * borders and set to zero.
	 * @param input Laplacian filtered Sinogram
	 */
	public void applyToGrid(Grid3D input) {
		double eps = 0.1;
		int maxProj = input.getSize()[2];
		int maxU = input.getSize()[0];
		int maxV = input.getSize()[1];
		int counter[][] = new int[maxV][maxU];

		for (int p = 0; p < maxProj; p++) {
			//left border
			for (int v = 0; v < maxV; v++) {
				for (int u = 0; u < maxU / 2; u++) {
					if (input.getAtIndex(u, v, p) * -1.0 > eps) {
						counter[v][u]++;
						break;
					}
				}
			}
			//top border
			for (int u = 0; u < maxU; u++) {
				for (int v = 0; v < maxV / 2; v++) {
					if (input.getAtIndex(u, v, p) * -1.0 > eps) {
						counter[v][u]++;
						break;
					}
				}
			}
			//right border
			for (int v = 0; v < maxV; v++) {
				for (int u = maxU - 1; u > maxU / 2; u--) {
					if (input.getAtIndex(u, v, p) * -1.0 > eps) {
						counter[v][u]++;
						break;
					}
				}
			}
			//bottom border
			for (int u = 0; u < maxU; u++) {
				for (int v = maxV - 1; v > maxV / 2; v--) {
					if (input.getAtIndex(u, v, p) * -1.0 > eps) {
						counter[v][u]++;
						break;
					}
				}
			}
		}
		int horizontal[] = new int[maxU];
		int vertical[] = new int[maxV];
		for (int i = 0; i < maxU; i++) {
			for (int j = 0; j < maxV; j++) {
				horizontal[i] += counter[j][i];
				vertical[j] += counter[j][i];
			}
		}
		//left
		int tmp1 = 0;
		int max1 = 0;
		for (int i = 0; i < maxU / 2; i++) {
			if (horizontal[i] > max1) {
				max1 = horizontal[i];
				tmp1 = i;
			}
		}
		//right
		int tmp2 = 0;
		int max2 = 0;
		for (int j = maxU - 1; j > maxU / 2; j--) {
			if (horizontal[j] > max2) {
				max2 = horizontal[j];
				tmp2 = j;
			}
		}
		//top
		int tmp3 = 0;
		int max3 = 0;
		for (int i = 0; i < maxV / 2; i++) {
			if (vertical[i] > max3) {
				max3 = vertical[i];
				tmp3 = i;
			}
		}
		//bottom
		int tmp4 = 0;
		int max4 = 0;
		for (int j = maxV - 1; j > maxV / 2; j--) {
			if (vertical[j] > max4) {
				max4 = vertical[j];
				tmp4 = j;
			}
		}

		for (int i = 0; i < maxProj; i++) {
			for (int v = 0; v < maxV; v++) {
				input.setAtIndex(tmp1, v, i, 0);
				input.setAtIndex(tmp1 + 1, v, i, 0);
				input.setAtIndex(tmp2, v, i, 0);
				input.setAtIndex(tmp2 - 1, v, i, 0);
			}
			for (int u = 0; u < maxU; u++) {
				input.setAtIndex(u, tmp3, i, 0);
				input.setAtIndex(u, tmp3 + 1, i, 0);
				input.setAtIndex(u, tmp4, i, 0);
				input.setAtIndex(u, tmp4 - 1, i, 0);
			}

		}

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
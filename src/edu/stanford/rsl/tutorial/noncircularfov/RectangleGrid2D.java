/*
 * Copyright (C) 2014 Madgalena Herbst
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.noncircularfov;

import edu.stanford.rsl.tutorial.phantoms.Phantom;

/**
 * 
 * @author Magdalena
 * 
 */
public class RectangleGrid2D extends Phantom {

	/**
	 * The constructor takes four arguments to initialize the grid. The
	 * rectangle will be in the center and have the size 2*b x 2*a.
	 * 
	 * @param x
	 *            no. of pixel in x-direction
	 * @param y
	 *            no. of pixel in y-direction
	 * @param a
	 * @param b
	 */
	public RectangleGrid2D(int x, int y, double a, double b) {
		super(x, y, "RectangleGrid2D");

		// Create rectangle in the grid.
		double radiusA = a * x;
		double radiusB = b * x;
		int xcenter = x / 2;
		int ycenter = y / 2;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.abs(i - xcenter) <= radiusA
						&& Math.abs(j - ycenter) <= radiusB) {
					super.setAtIndex(i, j, val);
				}
			}
		}
	}
}

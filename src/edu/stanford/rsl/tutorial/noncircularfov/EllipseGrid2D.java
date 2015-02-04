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
public class EllipseGrid2D extends Phantom {

	/**
	 * The constructor takes four arguments to initialize the grid. The ellipse
	 * will be in the center and have a radius in x-direction of a and in
	 * y-direction of b.
	 * 
	 * @param x width of the phantom
	 * @param y height of the phantom
	 * @param a ellipse radius
	 * @param b ellipse radius
	 */
	public EllipseGrid2D(int x, int y, double a, double b) {
		super(x, y, "EllipseGrid2D");

		// Create ellipse in the grid.
		double radiusA = a * x;
		double radiusB = b * x;
		int xcenter = x / 2;
		int ycenter = y / 2;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.pow(i - xcenter, 2) / (radiusA * radiusA)
						+ Math.pow(j - ycenter, 2) / (radiusB * radiusB) <= 1) {
					super.setAtIndex(i, j, val);
				}
			}
		}
	}
}

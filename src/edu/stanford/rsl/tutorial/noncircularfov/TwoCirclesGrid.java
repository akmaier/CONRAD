package edu.stanford.rsl.tutorial.noncircularfov;

/*
 * Copyright (C) 2014 Madgalena Herbst
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
import edu.stanford.rsl.tutorial.phantoms.Phantom;

/**
 * 
 * @author Magdalena
 * 
 */
public class TwoCirclesGrid extends Phantom {

	/**
	 * creates a phantom that consists of two circles with radius = 0.15*width
	 * 
	 * @param x
	 *            width of the phantom
	 * @param y
	 *            height of the phantom
	 */
	public TwoCirclesGrid(int x, int y) {

		super(x, y, "TwoCirclesGrid");

		double radius = 0.15 * x;
		int xcenter1 = (int) (x / 4 + x * 0.05);
		int ycenter1 = (int) (y * 0.5);

		int xcenter2 = (int) ((x * 0.75) - x * 0.05);
		int ycenter2 = (int) (y * 0.5);
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.pow(i - xcenter1, 2) + Math.pow(j - ycenter1, 2) <= (radius * radius)
						|| Math.pow(i - xcenter2, 2)
								+ Math.pow(j - ycenter2, 2) <= (radius * radius)) {
					super.setAtIndex(i, j, val);
				}
			}
		}
	}
}

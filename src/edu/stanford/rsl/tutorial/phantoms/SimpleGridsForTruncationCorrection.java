package edu.stanford.rsl.tutorial.phantoms;

public class SimpleGridsForTruncationCorrection extends Phantom {
	
	/**
	 * The constructor takes three arguments to initialize the grid. The geometry
	 * will be in the center. Thus a grid size that is uniform in both directions
	 * is recommended.
	 * 
	 * @param x:		image size in x-direction
	 * @param y: 		image size in y-direction
	 * @param geometry:	0: circle
	 * 					1: ellipse
	 * 					2: square
	 */
	
	public SimpleGridsForTruncationCorrection(int x, int y, int geometry) {
		super(x, y, "UniformCircleGrid2D");
		
		if (geometry == 0) {

			// Create circle in the grid.
			double radius = 0.4 * x;
			int xcenter = x / 2;
			int ycenter = y / 2;

			float val = 1.f;

			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++) {
					if (Math.pow((i - xcenter), 2) + Math.pow((j - ycenter), 2) <= (radius * radius)) {
						super.setAtIndex(i, j, val);
					}
				}
			}
			
		} else if (geometry == 1) {
			// Create ellipse in the grid.
			double radius = 0.2 * x;
			int xcenter = x / 2;
			int ycenter = y / 2;

			float val = 1.f;

			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++) {
					if (Math.pow((i - xcenter)/2, 2) + Math.pow((j - ycenter), 2) <= (radius * radius)) {
						super.setAtIndex(i, j, val);
					}
				}
			}
		} else if (geometry == 2) {
			// Create square in the grid.
			double radius = 0.35 * x;
			int xcenter = x / 2;
			int ycenter = y / 2;
			float val = 1.f;

			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++) {
					if (i <= radius+xcenter && j <= radius+ycenter && i > xcenter-radius && j > ycenter-radius ) {
						super.setAtIndex(i, j, val);
					}
				}
			}
		}
		
	}

}

/*
 * Copyright (C) 2015 Jennifer Maier, jennifer.maier@fau.de
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

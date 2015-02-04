package edu.stanford.rsl.tutorial.phantoms;

/**
 * Simple class to show the Grid2D functionality. We use a Grid2D to create a
 * uniform circle. We use this as a first phantom to investigate parallel beam
 * projection and back-projection.
 * 
 * @author Recopra Summer Semester 2012
 * 
 */
public class UniformCircleGrid2D extends Phantom {

	/**
	 * The constructor takes two arguments to initialize the grid. The circle
	 * will be in the center and have a radius of 45% of the x-dimension. Thus
	 * we recommend a grid size that is uniform in both directions.
	 * 
	 * @param x
	 * @param y
	 */
	public UniformCircleGrid2D(int x, int y) {
		super(x, y, "UniformCircleGrid2D");

		// Create circle in the grid.
		double radius = 0.4 * x;
		int xcenter = x / 2;
		int ycenter = y / 2;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
					super.setAtIndex(i, j, val);
				}
			}
		}

	}

	/**
	 * The constructor takes two arguments to initialize the grid. The circle
	 * will be in the center and have a radius of r% of the x-dimension. Thus
	 * we recommend a grid size that is uniform in both directions.
	 * 
	 * @param x
	 * @param y
	 */
	public UniformCircleGrid2D(int x, int y, double r) {
		super(x, y, "UniformCircle: Radius "+Double.toString(r*x));

		// Create circle in the grid.
		double radius = r * x;
		int xcenter = x / 2;
		int ycenter = y / 2;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
					super.setAtIndex(i, j, val);
				}
			}
		}

	}
	
	/**
	 * The constructor takes two arguments to initialize the grid. The circle
	 * will be in the center and have a radius of r% of the x-dimension. Thus
	 * we recommend a grid size that is uniform in both directions.
	 * 
	 * @param x
	 * @param y
	 */
	public UniformCircleGrid2D(int x, int y, int offx, int offy, double r) {
		super(x, y, "UniformCircle: Radius "+Double.toString(r*x));

		// Create circle in the grid.
		double radius = r * x;
		int xcenter = x / 2 + offx;
		int ycenter = y / 2 + offy;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
					super.setAtIndex(i, j, val);
				}
			}
		}

	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
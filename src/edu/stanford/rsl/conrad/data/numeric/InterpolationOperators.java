/*
 * Copyright (C) 2010-2014 - Andreas Maier, Andreas Keil 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;


/** The collection of all interpolation operators for Grids. */
public abstract class InterpolationOperators {


	///////////////////////////////////////////////////////////////////////
	// Linear interpolation of Grids                                     //
	///////////////////////////////////////////////////////////////////////

	/** Linear interpolation of a 1D Grid */
	public static float interpolateLinear(final Grid1D grid, double i) {
		int lower = (int) Math.floor(i);
		double d = i - lower; // d is in [0, 1)
		return (float) (
				(1.0-d)*grid.getAtIndex(lower)
				+ ((d != 0.0) ? d*grid.getAtIndex(lower+1) : 0.0)
		);
	}

	/** Linear interpolation of a 2D Grid */
	public static float interpolateLinear(final Grid2D grid, double x, double y) {
		if (grid == null) return 0;
		if (x < 0 || x > grid.getSize()[0]-1 || y < 0 || y > grid.getSize()[1]-1)
			return 0;
		
		int lower = (int) Math.floor(y);
		double d = y - lower; // d is in [0, 1)

		return (float) (
				(1.0-d)*interpolateLinear(grid.getSubGrid(lower), x)
				+ ((d != 0.0) ? d*interpolateLinear(grid.getSubGrid(lower+1), x) : 0.0)
		);
	}

	/** Linear interpolation of a 3D Grid */
	public static float interpolateLinear(final Grid3D grid, double z, double x, double y) {
		int lower = (int) Math.floor(z);
		double d = z - lower; // d is in [0, 1)
		return (float) (
				(1.0-d)*interpolateLinear(grid.getSubGrid(lower), x, y)
				+ ((d != 0.0) ? d*interpolateLinear(grid.getSubGrid(lower+1), x, y) : 0.0)
		);
	}


	///////////////////////////////////////////////////////////////////////
	// Interpolated updates into Grids                                   //
	// (These methods "spread out" an additive update to a grid using    //
	// the same weighting factors as for interpolation.)                 //
	///////////////////////////////////////////////////////////////////////

	/** Linear extrapolation into a 1D Grid */
	public static void addInterpolateLinear(final Grid1D grid, double x, float val) {
		int lower = (int) Math.floor(x);
		double d = x - lower;
		grid.setAtIndex(lower, grid.getAtIndex(lower) + (float) ((1.0-d)*val));
		if (d != 0.0) grid.setAtIndex(lower+1, grid.getAtIndex(lower+1) + (float) (d*val));
	}

	/** Linear extrapolation into a 2D Grid */
	public static void addInterpolateLinear(final Grid2D grid, double x, double y, float val) {
		int lower = (int) Math.floor(y);
		double d = y - lower;
		addInterpolateLinear(grid.getSubGrid(lower), x, (float) ((1.0-d)*val));
		if (d != 0.0) addInterpolateLinear(grid.getSubGrid(lower+1), x, (float) (d*val));
	}
	
	/** Linear extrapolation into a 3D Grid */
	public static void addInterpolateLinear(final Grid3D grid, double x, double y, double z, float val) {
		int lower = (int) Math.floor(z);
		double d = z - lower;
		addInterpolateLinear(grid.getSubGrid(lower), x, y, (float) ((1.0-d)*val));
		if (d != 0.0) addInterpolateLinear(grid.getSubGrid(lower+1), x, y, (float) (d*val));
	}
}


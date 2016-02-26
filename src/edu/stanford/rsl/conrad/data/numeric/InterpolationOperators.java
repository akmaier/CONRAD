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

	public static enum boundaryHandling {
		ZEROPAD, REPLICATE
	}
	
	private static Integer correctBoundsOfIndex(int idx, int gridSize, boundaryHandling bounds){
		Integer retVal = idx;
		switch (bounds) {
		case ZEROPAD:
			if (idx < 0 || idx > gridSize-1)
				retVal = null;
			break;
		case REPLICATE:
			if (idx < 0)
				retVal = 0;
			if (idx > gridSize-1)
				retVal = gridSize-1;
			break;
		default:
			if (idx < 0 || idx > gridSize-1)
				retVal = null;
			break;
		}
		return retVal;
	}

	/** Linear interpolation of a 1D Grid */
	public static float interpolateLinear(final Grid1D grid, double i){
		return interpolateLinear(grid, i, boundaryHandling.ZEROPAD);
	}

	/** Linear interpolation of a 1D Grid */
	public static float interpolateLinear(final Grid1D grid, double i, boundaryHandling bounds) {
		Integer lower = (int) Math.floor(i);
		double d = i - lower; // d is in [0, 1)
		Integer lowerPlusOne = lower+1;
		
		lower = correctBoundsOfIndex(lower, grid.getSize()[0], bounds);
		lowerPlusOne = correctBoundsOfIndex(lowerPlusOne, grid.getSize()[0], bounds);

		return (float) (
				(1.0-d)*((lower!=null) ? grid.getAtIndex(lower) : 0.0)
				+ ((d != 0.0) ? d*((lowerPlusOne != null) ? grid.getAtIndex(lowerPlusOne): 0.0) : 0.0)
				);
	}
	
	
	/** Linear interpolation of a 2D Grid */
	public static float interpolateLinear(final Grid2D grid, double x, double y){
		return interpolateLinear(grid, x, y, boundaryHandling.ZEROPAD);
	}
	
	/** Linear interpolation of a 2D Grid */
	public static float interpolateLinear(final Grid2D grid, double x, double y, boundaryHandling bounds) {
		if (grid == null) return 0;
		
		Integer lower = (int) Math.floor(y);
		double d = y - lower; // d is in [0, 1)
		Integer lowerPlusOne = lower+1;
		
		lower = correctBoundsOfIndex(lower, grid.getSize()[1], bounds);
		lowerPlusOne = correctBoundsOfIndex(lowerPlusOne, grid.getSize()[1], bounds);
			
		return (float) (
				(1.0-d)*((lower!=null) ? interpolateLinear(grid.getSubGrid(lower), x, bounds) : 0.0)
				+ ((d != 0.0) ? d*((lowerPlusOne != null) ? interpolateLinear(grid.getSubGrid(lowerPlusOne), x, bounds) : 0.0) : 0.0)
				);
	}

	
	/** Linear interpolation of a 3D Grid */
	public static float interpolateLinear(final Grid3D grid, double z, double x, double y){
		return interpolateLinear(grid, z, x, y, boundaryHandling.ZEROPAD);
	}
	
	/** Linear interpolation of a 3D Grid */
	public static float interpolateLinear(final Grid3D grid, double z, double x, double y, boundaryHandling bounds) {
		Integer lower = (int) Math.floor(z);
		double d = z - lower; // d is in [0, 1)
		Integer lowerPlusOne = lower+1;
		
		lower = correctBoundsOfIndex(lower, grid.getSize()[2], bounds);
		lowerPlusOne = correctBoundsOfIndex(lowerPlusOne, grid.getSize()[2], bounds);
		
		return (float) (
				(1.0-d)*((lower!=null) ? interpolateLinear(grid.getSubGrid(lower), x, y, bounds) : 0.0)
				+ ((d != 0.0) ? d*((lowerPlusOne != null) ? interpolateLinear(grid.getSubGrid(lowerPlusOne), x, y, bounds) : 0.0) : 0.0)
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


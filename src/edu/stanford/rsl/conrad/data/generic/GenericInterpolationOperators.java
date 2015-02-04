/*
 * Copyright (C) 2010-2014 - Andreas Maier, Andreas Keil 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.generic;

import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;


/** The collection of all interpolation operators for Grids. */
public abstract class GenericInterpolationOperators<T extends Gridable<T>> {


	///////////////////////////////////////////////////////////////////////
	// Linear interpolation of Grids                                     //
	///////////////////////////////////////////////////////////////////////

	protected abstract T getDefault();

	/** Linear interpolation of an ND Grid */
	public T interpolateLinear(final GenericGrid<T> grid, int currDim, double... idx) {
		int lower = (int) Math.floor(idx[currDim]);
		double d = idx[currDim] - lower; // d is in [0, 1)
		if(currDim<=0){
			int[] lowIdx = new int[idx.length];
			int[] highIdx = new int[idx.length];
			for (int i = 1; i < highIdx.length; i++) {
				lowIdx[i] = (int)idx[i];
				highIdx[i] = (int)idx[i];
			}
			lowIdx[0]=lower;
			highIdx[0]=lower+1;
			return grid.getValue(lowIdx).mul(1.0-d)
					.add((d != 0.0) ? grid.getValue(highIdx).mul(d) : getDefault());
		}

		double[] lowIdx = new double[idx.length];
		double[] highIdx = new double[idx.length];
		System.arraycopy(idx, 0, lowIdx, 0, idx.length);
		System.arraycopy(idx, 0, highIdx, 0, idx.length);
		lowIdx[currDim]=lower;
		highIdx[currDim]=lower+1;

		return interpolateLinear(grid, currDim-1, lowIdx).mul((1.0-d))
				.add((d != 0.0) ? interpolateLinear(grid, currDim-1, highIdx).mul(d) : getDefault());
	}
	
	/** Linear interpolation of an ND Grid */
	public T interpolateLinear(final GenericGrid<T> grid, double... idx) {
		return interpolateLinear(grid, idx.length-1, idx);
	}
	
	/** Linear interpolation of an ND Grid */
	public void addInterpolateLinear(final GenericGrid<T> grid, int currDim, T val, double... idx) {
		int lower = (int) Math.floor(idx[currDim]);
		double d = idx[currDim] - lower; // d is in [0, 1)
		
		if(currDim<=0){
			int[] lowIdx = new int[idx.length];
			int[] highIdx = new int[idx.length];
			for (int i = 1; i < highIdx.length; i++) {
				lowIdx[i] = (int)idx[i];
				highIdx[i] = (int)idx[i];
			}
			lowIdx[0]=lower;
			highIdx[0]=lower+1;
			grid.setValue(grid.getValue(lowIdx).add(val.mul(1.0-d)),lowIdx);
			if (d != 0.0) grid.setValue(grid.getValue(highIdx).add(val.mul(d)));
		}

		double[] lowIdx = new double[idx.length];
		double[] highIdx = new double[idx.length];
		System.arraycopy(idx, 0, lowIdx, 0, idx.length);
		System.arraycopy(idx, 0, highIdx, 0, idx.length);
		lowIdx[currDim]=lower;
		highIdx[currDim]=lower+1;

		addInterpolateLinear(grid, currDim-1, val.mul((1.0-d)), lowIdx);
		if (d != 0.0) addInterpolateLinear(grid, currDim-1, val.mul(d),highIdx);
	}
	
	/** Linear extrapolation into an ND Grid */
	public void addInterpolateLinear(final GenericGrid<T> grid, T val, double... idx) {
		addInterpolateLinear(grid, idx.length-1, val, idx);
	}
}


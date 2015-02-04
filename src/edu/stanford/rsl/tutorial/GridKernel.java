package edu.stanford.rsl.tutorial;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;

/**
 * Interface to describe a kernel that is applied row-wise to a sinogram.
 * @author akmaier
 *
 */
public interface GridKernel {
	/**
	 * Method to apply the kernel to a 1DGrid
	 * @param input
	 */
	void applyToGrid(Grid1D input);
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
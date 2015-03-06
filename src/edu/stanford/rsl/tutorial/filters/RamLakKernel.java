package edu.stanford.rsl.tutorial.filters;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Spatial discrete realization of the ramp filter using the Ram-Lak Convolver.
 * See L. Zeng. "Medical Image Reconstruction: A Conceptual tutorial". 2009, page 44
 * @author akmaier
 *
 */
public class RamLakKernel extends Grid1DComplex implements GridKernel {

	public RamLakKernel(final int size, double deltaS) {
		super(size);
		final int paddedSize = getSize()[0];
		final float odd = -1.f / ((float) (Math.PI * Math.PI * deltaS));
		setAtIndex(0, (float) (0.25f / (deltaS)));
		for (int i = 1; i < paddedSize/2; ++i) {
			if (1 == (i % 2))
				setAtIndex(i, odd / (i * i));
		}
		for (int i = paddedSize / 2; i < paddedSize; ++i) {
			final float tmp = paddedSize - i;
			if (1 == (tmp % 2))
				setAtIndex(i, odd / (tmp * tmp));
		}
		transformForward();
	}

	public void applyToGrid(Grid1D input) {

		Grid1DComplex subGrid = new Grid1DComplex(input);

		subGrid.transformForward();
		for (int idx = 0; idx < subGrid.getSize()[0]; ++idx) {
			subGrid.multiplyAtIndex(idx, getRealAtIndex(idx),
					getImagAtIndex(idx));
		}
		subGrid.transformInverse();

		Grid1D filteredSinoSub = subGrid.getRealSubGrid(0, input.getSize()[0]);
		NumericPointwiseOperators.copy(input, filteredSinoSub);

	}

	public final static void main(String[] args) {
		RamLakKernel r = new RamLakKernel(320, 2);
		VisualizationUtil.createPlot(r.getSubGrid(0, 512).getBuffer()).show();
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
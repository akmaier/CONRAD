package edu.stanford.rsl.tutorial.motion.estimation;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.filters.GridKernel;

/**
 * This class implements the 1-D version of a smoothing kernel (1 1 1)
 * @author Marco Boegel
 *
 */
public class SmoothKernel1D extends Grid1DComplex implements GridKernel {
	private int width = 0;

	/**
	 * 
	 * @param size
	 * @param width odd number!
	 */
	public SmoothKernel1D(final int size, int width) {
		super(FFTUtil.getNextPowerOfTwo(size));
		this.width = width;
		final int paddedSize = getSize()[0];

		for (int i = 0; i < paddedSize / 2; ++i) {
			setAtIndex(i, 0);
		}
		for (int i = paddedSize / 2; i < paddedSize; ++i) {
			setAtIndex(i, 0);
		}
		for (int i = 0; i <= width / 2; i++) {
			setAtIndex(i, 1.f / width);
		}
		for (int i = 1; i <= width / 2; i++) {
			setAtIndex(paddedSize - i, 1.f / width);
		}
		transformForward();
	}

	public Grid1D paddGrid(Grid1D input, int width) {
		Grid1D out = new Grid1D(input.getSize()[0] + width);
		int size = input.getSize()[0];
		for (int i = 0; i < width / 2; i++) {
			out.setAtIndex(i, input.getAtIndex(0));
		}
		for (int i = width / 2; i < width / 2 + size; i++) {
			out.setAtIndex(i, input.getAtIndex(i - width / 2));
		}
		for (int i = width / 2 + size; i < size + width; i++) {
			out.setAtIndex(i, input.getAtIndex(size - 1));
		}

		return out;
	}

	/**
	 * This method implements the Convolution in Fourier Space for a 1-D Grid.
	 * @param input 1-D Image
	 */
	public void applyToGrid(Grid1D input) {

		Grid1D paddedInput = paddGrid(input, width);
		Grid1DComplex subGrid = new Grid1DComplex(paddedInput);
		int size = subGrid.getSize()[0];
		subGrid.transformForward();
		for (int idx = 0; idx < size; ++idx) {
			subGrid.multiplyAtIndex(idx, getRealAtIndex(idx), getImagAtIndex(idx));
		}
		subGrid.transformInverse();

		Grid1D filteredSinoSub = subGrid.getRealSubGrid(width / 2, input.getSize()[0]);
		for (int i = 0; i < input.getSize()[0]; i++) {
			input.setAtIndex(i, filteredSinoSub.getAtIndex(i));
		}

	}

	/**
	 * This method implements the Convolution with a 2-D Image by applying the Filter to each 1-D subgrid.
	 * @param input 2-D Image
	 */
	public void applyToGrid(Grid2D input) {

		int iter = input.getSize()[1];

		for (int i = 0; i < iter; i++) {
			applyToGrid(input.getSubGrid(i));
		}
	}

	/**
	 * This method implements the Convolution with a 3-D Image by applying the Filter recursively to each 2-D subgrid.
	 * @param input 3-D Image
	 */
	public void applyToGrid(Grid3D input) {

		int iter = input.getSize()[2];

		for (int i = 0; i < iter; i++) {
			applyToGrid(input.getSubGrid(i));
		}
	}

	public final static void main(String[] args) {
		final int size = 200;
		SmoothKernel1D r = new SmoothKernel1D(size, 9);
		Grid1D in = new Grid1D(new float[size]);

		for (int i = 0; i < size; i++)
			in.setAtIndex(i, 0);
		int k = 30;
		for (int i = 0; i < size / 2; i++) {
			in.setAtIndex(i, k);
		}
		for (int i = size / 2; i < 5 * size / 8; i++) {
			in.setAtIndex(i, k);
		}
		VisualizationUtil.createPlot("before", in.getBuffer()).show();
		VisualizationUtil.createPlot("Filter", r.getSubGrid(0, FFTUtil.getNextPowerOfTwo(size)).getBuffer()).show();
		r.applyToGrid(in);
		VisualizationUtil.createPlot("after", in.getBuffer()).show();

	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

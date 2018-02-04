package edu.stanford.rsl.tutorial.atract;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.filters.GridKernel;

/**
 * This class implements the 1-D version of the Laplacian Kernel ( 1 -2 1 )
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class LaplaceKernel1D extends Grid1DComplex implements GridKernel {

	public LaplaceKernel1D(final int size) {
		// initialize the superclass
		super(size);
		// gets the real size (Number of complex numbers)
		final int paddedSize = getSize()[0];

		// set the kernel
		setAtIndex(0, -2);
		setAtIndex(1, 1);
		setAtIndex(paddedSize - 1, 1);

		// do forward fourier transform
		transformForward();
	}

	/**
	 * This method implements the Convolution in Fourier Space for a 1-D Grid.
	 * @param input 1-D Image
	 */
	public void applyToGrid(Grid1D input) {

		Grid1DComplex subGrid = new Grid1DComplex(input);
		int size = subGrid.getSize()[0];
		subGrid.transformForward();
		for (int idx = 0; idx < size; ++idx) {
			subGrid.multiplyAtIndex(idx, getRealAtIndex(idx), getImagAtIndex(idx));
		}
		subGrid.transformInverse();

		Grid1D filteredSinoSub = subGrid.getRealSubGrid(0, input.getSize()[0]);
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
		new ImageJ();
		// 1D example
		final int size = 200;
		LaplaceKernel1D r = new LaplaceKernel1D(size);
		Grid1D in = new Grid1D(new float[size]);

		for (int i = 0; i < size; i++)
			in.setAtIndex(i, 0);
		int k = 30;
		for (int i = 3 * size / 8; i < size / 2; i++) {
			in.setAtIndex(i, k);
		}
		for (int i = size / 2; i < 5 * size / 8; i++) {
			in.setAtIndex(i, k);
		}
		VisualizationUtil.createPlot("before", in.getBuffer()).show();
		VisualizationUtil.createPlot("Filter", r.getMagSubGrid(0, r.getSize()[0]).getBuffer()).show();
		// save old signal
		Grid1D inSave = new Grid1D(in);
		r.applyToGrid(in);
		VisualizationUtil.createPlot("after", in.getBuffer()).show();

		// 2D example
		Grid2D in2D = new Grid2D(size, size);
		for (int j = 0; j < in2D.getHeight(); j++) {
			for (int i = 0; i < in2D.getWidth(); i++) {
				in2D.setAtIndex(i, j, inSave.getAtIndex(i));
			}
		}

		in2D.show("Grid2D input");
		r.applyToGrid(in2D);
		in2D.show("Grid2D Laplace");

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

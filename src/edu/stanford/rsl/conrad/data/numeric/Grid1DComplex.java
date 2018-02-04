/*
 * Copyright (C) 2010-2014 - Andreas Maier, Martin Berger, Maro Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.stanford.rsl.conrad.utils.FFTUtil;

/**
 * Class to use complex numbers in a grid structure.
 * Internally float arrays are used. The real and imaginary parts are stored in an alternating manner.
 * (Note: OpenCL memory management is supported but pointwise operations on the GPU will not execute complex math)
 * @author akmaier / berger / boegel
 *
 */
public class Grid1DComplex extends Grid1D {

	public Grid1DComplex(int width) {
		this(width, true);
	}

	public Grid1DComplex(int width, boolean extendToNextPowerOfTwo) {
		super(extendToNextPowerOfTwo ? FFTUtil.getNextPowerOfTwo(width) * 2 : width * 2);
	}

	public Grid1DComplex(Grid1D grid) {
		this(grid, true);
	}

	public Grid1DComplex(Grid1D grid, boolean extendToNextPowerOfTwo) {
		super(extendToNextPowerOfTwo ? FFTUtil.getNextPowerOfTwo(grid.getSize()[0]) * 2 : grid.getSize()[0] * 2);
		final int inputSize = grid.getSize()[0];
		for (int i = 0; i < inputSize; ++i) {
			this.setAtIndex(i, grid.getAtIndex(i));
		}
	}

	public Grid1DComplex(Grid1DComplex grid) {
		this(grid, true);
	}

	public Grid1DComplex(Grid1DComplex grid, boolean extendToNextPowerOfTwo) {
		super(extendToNextPowerOfTwo ? FFTUtil.getNextPowerOfTwo(grid.getSize()[0]) * 2 : grid.getSize()[0] * 2);
		final int inputSize = grid.getSize()[0];
		for (int i = 0; i < inputSize; ++i) {
			this.setRealAtIndex(i, grid.getRealAtIndex(i));
			this.setImagAtIndex(i, grid.getImagAtIndex(i));
		}
	}

	public float getAtIndex(int index) {
		return (float) Math.sqrt(super.getAtIndex(index * 2) * super.getAtIndex(index * 2)
				+ super.getAtIndex(index * 2 + 1) * super.getAtIndex(index * 2 + 1));
	}

	@Override
	public void multiplyAtIndex(int index, float val) {
		super.multiplyAtIndex(index * 2, val);
		super.multiplyAtIndex(index * 2 + 1, val);
	}

	public void multiplyAtIndex(int index, float real, float imag) {
		float tmp_re = super.getAtIndex(index * 2);
		super.setAtIndex(index * 2, real * super.getAtIndex(index * 2) - imag * super.getAtIndex(index * 2 + 1));
		super.setAtIndex(index * 2 + 1, imag * tmp_re + real * super.getAtIndex(index * 2 + 1));
	}

	@Override
	public void addAtIndex(int index, float val) {
		super.addAtIndex(index * 2, val);
	}

	public void addAtIndex(int index, float real, float imag) {
		super.addAtIndex(index * 2, real);
		super.addAtIndex(index * 2 + 1, imag);
	}

	public void setAtIndex(int index, float val) {
		super.setAtIndex(index * 2, val);
		super.setAtIndex(index * 2 + 1, 0);
	}

	public void setAtIndex(int index, float real, float imag) {
		super.setAtIndex(index * 2, real);
		super.setAtIndex(index * 2 + 1, imag);
	}

	public float getRealAtIndex(int index) {
		return super.getAtIndex(index * 2);
	}

	public float getImagAtIndex(int index) {
		return super.getAtIndex(index * 2 + 1);
	}

	public void setRealAtIndex(int index, float val) {
		super.setAtIndex(index * 2, val);
	}

	public void setImagAtIndex(int index, float val) {
		super.setAtIndex(index * 2 + 1, val);
	}

	public void transformForward() {
		FloatFFT_1D fft = new FloatFFT_1D(getSize()[0]);
		fft.complexForward(this.buffer); // TODO: Only works if we do not have an offset in the 1D grid
	}

	public void transformInverse() {
		FloatFFT_1D fft = new FloatFFT_1D(getSize()[0]);
		fft.complexInverse(this.buffer, true); // TODO: Only works if we do not have an offset in the 1D grid
	}

	public Grid1D getRealSubGrid(final int startIndex, final int length) {
		Grid1D subgrid = new Grid1D(new float[length]);
		for (int i = 0; i < length; ++i) {
			subgrid.setAtIndex(i, super.getAtIndex((startIndex + i) * 2));
		}
		return subgrid;
	}

	public Grid1D getImagSubGrid(final int startIndex, final int length) {
		Grid1D subgrid = new Grid1D(new float[length]);
		for (int i = 0; i < length; ++i) {
			subgrid.setAtIndex(i, super.getAtIndex((startIndex + i) * 2 + 1));
		}
		return subgrid;
	}

	public Grid1D getMagSubGrid(final int startIndex, final int length) {
		Grid1D subgrid = new Grid1D(new float[length]);
		for (int i = 0; i < length; ++i) {
			float real = super.getAtIndex((startIndex + i) * 2);
			float imag = super.getAtIndex((startIndex + i) * 2 + 1);
			subgrid.setAtIndex(i, (float) Math.sqrt(real * real + imag * imag));
		}
		return subgrid;
	}

	public Grid1D getPhaseSubGrid(final int startIndex, final int length) {
		Grid1D subgrid = new Grid1D(new float[length]);
		for (int i = 0; i < length; ++i) {
			float real = super.getAtIndex((startIndex + i) * 2);
			float imag = super.getAtIndex((startIndex + i) * 2 + 1);
			subgrid.setAtIndex(i, (float) Math.atan(real / imag));
		}
		return subgrid;
	}

	public void show() {
		show("");
	}

	public void show(String title) {
		this.getRealSubGrid(0, this.getNumberOfElements() / 2).show(title + " Real Part");
		this.getImagSubGrid(0, this.getNumberOfElements() / 2).show(title + " Imaginary Part");
	}

	@Override
	public int[] getSize() {
		return new int[] { size[0] / 2 };
	}

}

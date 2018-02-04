/*
 * Copyright (C) 2010-2014 - Andreas Maier, Martin Berger, Maro Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import ij.ImageJ;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;

/**
 * Class to use complex numbers in a grid structure.
 * Internally float arrays are used. The real and imaginary parts are stored in an alternating manner.
 * GridStructure:
 * R I R I R I
 * R I R I R I
 * R I R I R I
 * Therefore the width is double the width of a normal Grid2D
 * (Note: OpenCL memory management is supported but pointwise operations on the GPU will not execute complex math)
 * @author Marco Boegel
 *
 */
public class Grid2DComplex extends Grid2D {

	public Grid2DComplex(int width, int height) {
		this(width, height, true);
	}

	public Grid2DComplex(int width, int height, boolean extendToNextPow2) {
		super(extendToNextPow2 ? FFTUtil.getNextPowerOfTwo(width) * 2 : width * 2,
				extendToNextPow2 ? FFTUtil.getNextPowerOfTwo(height) : height);
	}

	public Grid2DComplex(Grid2D grid, boolean extendToNextPow2) {
		this(grid.getSize()[0], grid.getSize()[1], extendToNextPow2);
		final int inputSize1 = grid.getSize()[0];
		final int inputSize2 = grid.getSize()[1];
		for (int j = 0; j < inputSize2; ++j) {
			for (int i = 0; i < inputSize1; ++i) {
				this.setAtIndex(i, j, grid.getAtIndex(i, j));
			}
		}
	}

	public Grid2DComplex(Grid2D grid) {
		this(grid.getSize()[0], grid.getSize()[1], true);
		final int inputSize1 = grid.getSize()[0];
		final int inputSize2 = grid.getSize()[1];
		for (int i = 0; i < inputSize1; ++i) {
			for (int j = 0; j < inputSize2; ++j) {
				this.setAtIndex(i, j, grid.getAtIndex(i, j));
			}
		}
	}

	public Grid2DComplex(Grid2DComplex grid) {
		super(grid.getSize()[0] * 2, grid.getSize()[1]);

		for (int i = 0; i < super.getNumberOfElements(); ++i)
			this.getBuffer()[i] = grid.getBuffer()[i];

	}

	@Override
	public Grid2DComplex clone() {
		return new Grid2DComplex(this);
	}

	/**
	 * implicitely returns the magnitude of the complex number
	 */
	public float getAtIndex(int i, int j) {
		return (float) Math
				.sqrt(Math.pow(getPixelValue(i * 2, j), 2) + Math.pow(getPixelValue(i * 2 + 1, j), 2));
	}

	@Override
	public void multiplyAtIndex(int i, int j, float val) {
		putPixelValue(i * 2, j, getPixelValue(i * 2, j) * val);
		putPixelValue(i * 2 + 1, j, getPixelValue(i * 2 + 1, j) * val);
	}

	public void multiplyAtIndex(int i, int j, float real, float imag) {
		float tmp_re = getPixelValue(i * 2, j);
		putPixelValue(i * 2, j, real * getPixelValue(i * 2, j) - imag * getPixelValue(i * 2 + 1, j));
		putPixelValue(i * 2 + 1, j, imag * tmp_re + real * getPixelValue(i * 2 + 1, j));
	}

	@Override
	public void addAtIndex(int i, int j, float val) {
		putPixelValue(i * 2, j, getPixelValue(i * 2, j) + val);
	}

	public void setAtIndex(int i, int j, float val) {
		putPixelValue(i * 2, j, val);
		putPixelValue(i * 2 + 1, j, 0);
	}

	public float getRealAtIndex(int i, int j) {
		return this.getPixelValue(i * 2, j);
	}

	public float getImagAtIndex(int i, int j) {
		return this.getPixelValue(i * 2 + 1, j);
	}

	public void setRealAtIndex(int i, int j, float val) {
		this.putPixelValue(i * 2, j, val);
	}

	public void setImagAtIndex(int i, int j, float val) {
		this.putPixelValue(i * 2 + 1, j, val);
	}

	public void transformForward() {
		if (getSize()[1] > 1) {
			FloatFFT_2D fft = new FloatFFT_2D(getSize()[1], getSize()[0]);
			fft.complexForward(this.getBuffer());
		} else {
			FloatFFT_1D fft = new FloatFFT_1D(getSize()[0]);
			fft.complexForward(this.getBuffer());
		}
	}

	public void transformInverse() {
		if (getSize()[1] > 1) {
			FloatFFT_2D fft = new FloatFFT_2D(getSize()[1], getSize()[0]);
			fft.complexInverse(this.getBuffer(), true);
		} else {
			FloatFFT_1D fft = new FloatFFT_1D(getSize()[0]);
			fft.complexInverse(this.getBuffer(), true);
		}
	}

	public Grid2D getRealSubGrid(final int startIndexX, final int startIndexY, final int lengthX,
			final int lengthY) {
		Grid2D subgrid = new Grid2D(lengthX, lengthY);
		for (int i = 0; i < lengthX; ++i) {
			for (int j = 0; j < lengthY; ++j) {
				subgrid.setAtIndex(i, j, getPixelValue(2 * (startIndexX + i), startIndexY + j));
			}
		}
		return subgrid;
	}

	public Grid2D getImagSubGrid(final int startIndexX, final int startIndexY, final int lengthX,
			final int lengthY) {
		Grid2D subgrid = new Grid2D(lengthX, lengthY);
		for (int i = 0; i < lengthX; ++i) {
			for (int j = 0; j < lengthY; ++j) {
				subgrid.setAtIndex(i, j, getPixelValue(2 * (startIndexX + i) + 1, startIndexY + j));
			}
		}
		return subgrid;
	}

	public Grid2D getMagnSubGrid(final int startIndexX, final int startIndexY, final int lengthX,
			final int lengthY) {
		Grid2D subgrid = new Grid2D(lengthX, lengthY);
		for (int i = 0; i < lengthX; ++i) {
			for (int j = 0; j < lengthY; ++j) {
				float val = (float) getAtIndex(i, j);
				subgrid.setAtIndex(i, j, val);
			}
		}
		return subgrid;
	}

	public Grid2D getPhaseSubGrid(final int startIndexX, final int startIndexY, final int lengthX,
			final int lengthY) {
		Grid2D subgrid = new Grid2D(lengthX, lengthY);
		for (int i = 0; i < lengthX; ++i) {
			for (int j = 0; j < lengthY; ++j) {
				float val = (float) Math.atan(getPixelValue((startIndexX + i) * 2 + 1,
						(startIndexY + j))
						/ getPixelValue((startIndexX + i) * 2, startIndexY + j));
				subgrid.setAtIndex(i, j, val);
			}
		}
		return subgrid;
	}

	public void fftshift() {
		Grid2DComplex copy = new Grid2DComplex(this);

		int[] dim = this.getSize();
		int[] halfDim = new int[2];

		if (dim[0] % 2 == 0 && dim[1] % 2 == 0) {
			halfDim[0] = (int) Math.ceil(dim[0] / 2);
			halfDim[1] = (int) Math.ceil(dim[1] / 2);
		} else if (dim[0] % 2 != 0 && dim[1] % 2 == 0) {
			halfDim[0] = (int) Math.ceil(dim[0] / 2 + 1);
			halfDim[1] = (int) Math.ceil(dim[1] / 2);
		} else if (dim[0] % 2 == 0 && dim[1] % 2 != 0) {
			halfDim[0] = (int) Math.ceil(dim[0] / 2);
			halfDim[1] = (int) Math.ceil(dim[1] / 2 + 1);
		} else {
			halfDim[0] = (int) Math.ceil(dim[0] / 2 + 1);
			halfDim[1] = (int) Math.ceil(dim[1] / 2 + 1);
		}

		// swap section 4 to 1
		for (int i = 0; i < dim[1] - halfDim[1]; ++i) {
			for (int j = 0; j < dim[0] - halfDim[0]; ++j) {
				this.setRealAtIndex(j, i, copy.getRealAtIndex(j + halfDim[0], i + halfDim[1]));
				this.setImagAtIndex(j, i, copy.getImagAtIndex(j + halfDim[0], i + halfDim[1]));
			}
		}

		// swap section 3 to 2
		for (int i = 0; i < dim[1] - halfDim[1]; ++i) {
			for (int j = dim[0] - halfDim[0]; j < dim[0]; ++j) {
				this.setRealAtIndex(j, i,
						copy.getRealAtIndex(j - (dim[0] - halfDim[0]), i + halfDim[1]));
				this.setImagAtIndex(j, i,
						copy.getImagAtIndex(j - (dim[0] - halfDim[0]), i + halfDim[1]));
			}
		}
		// swap section 2 to 3
		for (int i = dim[1] - halfDim[1]; i < dim[1]; ++i) {
			for (int j = 0; j < dim[0] - halfDim[0]; ++j) {
				this.setRealAtIndex(j, i,
						copy.getRealAtIndex(j + halfDim[0], i - (dim[1] - halfDim[1])));
				this.setImagAtIndex(j, i,
						copy.getImagAtIndex(j + halfDim[0], i - (dim[1] - halfDim[1])));
			}
		}

		// swap section 1 to 4
		for (int i = dim[1] - halfDim[1]; i < dim[1]; ++i) {
			for (int j = dim[0] - halfDim[0]; j < dim[0]; ++j) {
				this.setRealAtIndex(j, i, copy.getRealAtIndex(j - (dim[0] - halfDim[0]),
						i - (dim[1] - halfDim[1])));
				this.setImagAtIndex(j, i, copy.getImagAtIndex(j - (dim[0] - halfDim[0]),
						i - (dim[1] - halfDim[1])));
			}
		}
	}

	public void ifftshift() {
		Grid2DComplex copy = new Grid2DComplex(this);

		int[] dim = this.getSize();
		int[] halfDim = { (int) Math.floor(dim[0] / 2), (int) Math.floor(dim[1] / 2) };

		// swap section 4 to 1
		for (int i = 0; i < dim[1] - halfDim[1]; ++i) {
			for (int j = 0; j < dim[0] - halfDim[0]; ++j) {
				this.setRealAtIndex(j, i, copy.getRealAtIndex(j + halfDim[0], i + halfDim[1]));
				this.setImagAtIndex(j, i, copy.getImagAtIndex(j + halfDim[0], i + halfDim[1]));
			}
		}

		// swap section 3 to 2
		for (int i = 0; i < dim[1] - halfDim[1]; ++i) {
			for (int j = dim[0] - halfDim[0]; j < dim[0]; ++j) {
				this.setRealAtIndex(j, i,
						copy.getRealAtIndex(j - (dim[0] - halfDim[0]), i + halfDim[1]));
				this.setImagAtIndex(j, i,
						copy.getImagAtIndex(j - (dim[0] - halfDim[0]), i + halfDim[1]));
			}
		}
		// swap section 2 to 3
		for (int i = dim[1] - halfDim[1]; i < dim[1]; ++i) {
			for (int j = 0; j < dim[0] - halfDim[0]; ++j) {
				this.setRealAtIndex(j, i,
						copy.getRealAtIndex(j + halfDim[0], i - (dim[1] - halfDim[1])));
				this.setImagAtIndex(j, i,
						copy.getImagAtIndex(j + halfDim[0], i - (dim[1] - halfDim[1])));
			}
		}

		// swap section 1 to 4
		for (int i = dim[1] - halfDim[1]; i < dim[1]; ++i) {
			for (int j = dim[0] - halfDim[0]; j < dim[0]; ++j) {
				this.setRealAtIndex(j, i, copy.getRealAtIndex(j - (dim[0] - halfDim[0]),
						i - (dim[1] - halfDim[1])));
				this.setImagAtIndex(j, i, copy.getImagAtIndex(j - (dim[0] - halfDim[0]),
						i - (dim[1] - halfDim[1])));
			}
		}
	}

	@Override
	public int[] getSize() {
		return new int[] { size[0] / 2, size[1] };
	}

	@Override
	public int getWidth() {
		return super.getWidth() / 2;
	}

	@Override
	public void show(String title) {
		VisualizationUtil.showGrid2D(this.getMagnSubGrid(0, 0, getSize()[0], getSize()[1]), title);
	}

	@Override
	public void show() {
		show("Grid2DComplex");
	}

	@Override
	public int getNumberOfElements() {
		int tmp = getSize()[0];
		for (int i = 1; i < getSize().length; ++i)
			tmp *= getSize()[i];
		return tmp;
	}

	public static void main(String[] args) {
		new ImageJ();
		Grid2DComplex test = new Grid2DComplex(new SheppLogan(256));
		Grid2DComplex ffttest = new Grid2DComplex(test);
		test.show();
		test.transformForward();
		test.show();
		test.transformInverse();
		test.show();

		test.getRealSubGrid(0, 0, 256, 256).show();
		test.getImagSubGrid(0, 0, 256, 256).show();
		ffttest.getRealSubGrid(0, 0, 256, 256).show();
		ffttest.getImagSubGrid(0, 0, 256, 256).show();

		Grid2DComplex diff = new Grid2DComplex(test.getSize()[0], test.getSize()[1]);
		for (int i = 0; i < test.getSize()[0]; ++i) {
			for (int j = 0; j < test.getSize()[1]; ++j) {
				diff.setRealAtIndex(i, j, ffttest.getRealAtIndex(i, j) - test.getRealAtIndex(i, j));
				diff.setImagAtIndex(i, j, ffttest.getImagAtIndex(i, j) - test.getImagAtIndex(i, j));
			}
		}

		diff.show();

	}

	@Override
	public String toString() {
		String out = "[ ";
		for (int j = 0; j < getSize()[1]; j++) {
			out += "[ ";
			for (int i = 0; i < getSize()[0]; i++) {
				out += new Complex(this.getRealAtIndex(i, j), this.getImagAtIndex(i, j));
				if (i < getSize()[0] - 1)
					out += ", ";
			}
			out += " ]";
			if (j < getSize()[1] - 1)
				out += "; ";
		}
		out += " ]";
		return out;
	}

}

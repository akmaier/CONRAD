package edu.stanford.rsl.tutorial.atract;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * This class implements the 1-D version of the ATRACT kernel.
 * @author Marco Boegel
 *
 */
public class AtractKernel1D_test extends Grid1DComplex {

	public AtractKernel1D_test(final int size) {
		super(FFTUtil.getNextPowerOfTwo(size));
		// size is double the actual size, since both Imaginary and Real part are saved in the same array
		final int paddedSizeReal = getSize()[0];
		/*
		final float c = (float) (-1/(4*Math.PI*Math.PI));
		final float step = (float) (1.0/(deltaS*paddedSizeReal));
		
		setAtIndex(0, c/(step));
		//positive range of the kernel
		for(int i = 1; i < paddedSizeReal/2; i++) {
			setAtIndex(i, c/(i*step));
		}
		//negative range of the kernel
		for(int i = paddedSizeReal/2;i < paddedSizeReal; i++) {
			final float tmp = i -paddedSizeReal; // -1*(padd - i)
			setAtIndex(i, c/(tmp*step));
		}
		*/
		setAtIndex(0, (float) (1 / (2 * Math.PI * Math.PI) * Math.log(Math.PI * Math.PI * 0.05) - 0.395));
		for (int i = 1; i < paddedSizeReal / 2; i++) {
			setAtIndex(i, (float) (1 / (2 * Math.PI * Math.PI) * Math.log(Math.PI * Math.PI * (double) i) - 0.395));
		}
		for (int i = paddedSizeReal / 2; i < paddedSizeReal; i++) {
			final double tmp = (double) (paddedSizeReal - i);
			double log = Math.log(Math.PI * Math.PI * tmp);
			float val = (float) (1 / (2 * Math.PI * Math.PI) * log - 0.395);
			setAtIndex(i, val);
		}
		transformForward();
	}

	public void applyToGrid(Grid1D input) {

		Grid1DComplex subGrid = new Grid1DComplex(input);

		subGrid.transformForward();
		for (int idx = 0; idx < subGrid.getSize()[0]; ++idx) {
			subGrid.multiplyAtIndex(idx, getRealAtIndex(idx), getImagAtIndex(idx));
		}
		subGrid.transformInverse();

		Grid1D filteredSinoSub = subGrid.getRealSubGrid(0, input.getSize()[0]);
		for (int i = 0; i < input.getSize()[0]; i++) {
			input.setAtIndex(i, filteredSinoSub.getAtIndex(i));
		}

	}

	public void applyToGrid(Grid2D input) {

		int iter = input.getSize()[0];
		for (int i = 0; i < iter; i++) {
			applyToGrid(input.getSubGrid(i));
		}

		double sum = 0;
		for (int i = 0; i < 133; ++i)
			sum += input.getSubGrid(i).getAtIndex(0);
		sum /= 133.0;
		Grid1D tmp = new Grid1D(input.getSubGrid(0));

		for (int i = 0; i < iter; i++) {
			if (input.getSubGrid(i).getAtIndex(0) != 0) {
				float h = (float) sum - input.getSubGrid(i).getAtIndex(0);
				NumericPointwiseOperators.addBy(input.getSubGrid(i), h);
			}
		}
	}

	public void applyToGrid(Grid3D input) {

		int iter = input.getSize()[0];

		for (int i = 0; i < iter; i++) {
			applyToGrid(input.getSubGrid(i));
		}
	}

	public final static void main(String[] args) {
		AtractKernel1D_test r = new AtractKernel1D_test(200);
		VisualizationUtil.createPlot(r.getSubGrid(0, 512).getBuffer()).show();
	}
}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
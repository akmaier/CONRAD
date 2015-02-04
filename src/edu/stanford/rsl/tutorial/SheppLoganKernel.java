package edu.stanford.rsl.tutorial;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * 
 */
public class SheppLoganKernel extends Grid1DComplex implements GridKernel {

	public SheppLoganKernel(final int size, double deltaS) {
		super(size);
		final int paddedSize = getSize()[0];
//		final float odd = -1.f / ((float) (Math.PI * Math.PI * deltaS));
		setAtIndex(0, (float) (2 / (Math.PI*Math.PI*deltaS*deltaS)));
		for (int i = 1; i < paddedSize/2; ++i) {
				setAtIndex(i,(float) (-2.0 / (Math.PI*Math.PI*deltaS*deltaS*(4*i*i-1.0))));
		}
		for (int i = paddedSize / 2; i < paddedSize; ++i) {
			final float tmp = paddedSize - i;
				setAtIndex(i,(float) (-2.0 / (Math.PI*Math.PI*deltaS*deltaS*(4*tmp*tmp-1.0))));
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
		SheppLoganKernel r = new SheppLoganKernel(320, 2);
		VisualizationUtil.createPlot(r.getSubGrid(0, 512).getBuffer()).show();
	}

}

package edu.stanford.rsl.tutorial.filters;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class RamLakKernelLinux extends Grid1DComplex implements GridKernel {

	 public RamLakKernelLinux(final int size, double deltaS) {
		super(FFTUtil.getNextPowerOfTwo(size) * 4);
		final int paddedSize = getSize()[0];
		final float odd = -1.f / ((float) (Math.PI * Math.PI*deltaS*deltaS));
		setAtIndex(0, (float) (0.25f / (deltaS*deltaS)));
		for (int i = 1; i < paddedSize / 4; ++i) {
			if (1 == (i % 2))
				setAtIndex(i, odd /(i * i));
		}
		for (int i = paddedSize / 4; i < paddedSize / 2; ++i) {
			final float tmp = paddedSize / 2 - i;
			if (1 == (tmp % 2))
				setAtIndex(i, odd / (tmp * tmp));
		}
		transformForward();
	}
	

	public void applyToGrid(Grid1D input) {

		Grid1DComplex subGrid = new Grid1DComplex(input);

		subGrid.transformForward();
		for (int idx = 0; idx < subGrid.getSize()[0] / 2; ++idx) {
			subGrid.multiplyAtIndex(idx, getRealAtIndex(idx),
					getImagAtIndex(idx));
		}
		subGrid.transformInverse();

		Grid1D filteredSinoSub = subGrid.getRealSubGrid(0, input.getSize()[0]);
		System.arraycopy(filteredSinoSub.getBuffer(), 0, input.getBuffer(), 0,
				filteredSinoSub.getSize()[0]);

	} 


	public final static void main(String[] args) {
		RamLakKernel r = new RamLakKernel(100, 1);
		VisualizationUtil.createPlot(r.getSubGrid(0, 256).getBuffer()).show();
	}

}

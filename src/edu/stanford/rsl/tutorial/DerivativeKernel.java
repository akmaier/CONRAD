package edu.stanford.rsl.tutorial;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

/**
 * First part of the decomposition of the ramp filter into a first derivative and a Hilbert filter.
 * See L. Zeng. "Medical Image Reconstruction: A Conceptual tutorial". 2009, page 28
 * @author akmaier
 *
 */

public class DerivativeKernel implements GridKernel {
	
	private Grid1D dKernel;
	public DerivativeKernel() {
		// TODO Auto-generated constructor stub
		dKernel = new Grid1D(new float[2]);
		dKernel.setAtIndex(0, -1.f);
		dKernel.setAtIndex(1, 1.f);
	}

	public void applyToGrid(Grid1D input) {
		float[] inputFloat = input.getBuffer();
		ImageProcessor ip = new FloatProcessor(inputFloat.length, 1, inputFloat);
		Convolver c = new Convolver();
		c.convolveFloat(ip, dKernel.getBuffer(), 2, 1);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Grid1D data = new Grid1D(new float[100]);
		
		for (int i=0;i<data.getSize()[0];++i) data.setAtIndex(i, (float) i);
		DerivativeKernel dKern = new DerivativeKernel();
		dKern.applyToGrid(data);
		VisualizationUtil.createPlot(data.getBuffer()).show();
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
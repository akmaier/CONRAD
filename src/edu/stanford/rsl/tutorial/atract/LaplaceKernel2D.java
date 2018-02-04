package edu.stanford.rsl.tutorial.atract;

import ij.ImageJ;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * This class implements the 2-D version of the Laplacian Kernel 
 * ( 0  -1  0 )
 * (-1   4 -1 )
 * ( 0  -1  0 )
 * 
 * The 2D Atract filter requires this filter kernel as opposed to the inverse kernel.
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class LaplaceKernel2D {
	private Grid2D dKernel;

	public LaplaceKernel2D() {
		dKernel = new Grid2D(3, 3);
		dKernel.setAtIndex(0, 0, 0.f);
		dKernel.setAtIndex(0, 1, -1.f);
		dKernel.setAtIndex(0, 2, 0.f);
		dKernel.setAtIndex(1, 0, -1.f);
		dKernel.setAtIndex(1, 1, 4.f);
		dKernel.setAtIndex(1, 2, -1.f);
		dKernel.setAtIndex(2, 0, 0.f);
		dKernel.setAtIndex(2, 1, -1.f);
		dKernel.setAtIndex(2, 2, 0.f);
	}

	public void applyToGrid(Grid2D input) {
		Convolver c = new Convolver();
		c.convolveFloat(ImageUtil.wrapGrid2D(input), dKernel.getBuffer(), 3, 3);
	}

	public void applyToGrid(Grid3D input) {

		int iter = input.getSize()[2];

		for (int i = 0; i < iter; i++) {
			applyToGrid(input.getSubGrid(i));
		}
	}

	public static void main(String[] args) {
		new ImageJ();

		int size = 200;
		// 2D example
		Grid2D in2D = new Grid2D(size, size);
		for (int j = 0; j < in2D.getHeight(); j++) {
			for (int i = 0; i < in2D.getWidth(); i++) {
				if (Math.sqrt(Math.pow((i - size / 2), 2) + Math.pow((j - size / 2), 2)) < 0.25 * size)
					in2D.setAtIndex(i, j, 30);
			}
		}

		LaplaceKernel2D kernel = new LaplaceKernel2D();
		in2D.show("Before");
		kernel.applyToGrid(in2D);
		in2D.show("After");

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
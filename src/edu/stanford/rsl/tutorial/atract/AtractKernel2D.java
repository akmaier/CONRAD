package edu.stanford.rsl.tutorial.atract;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.RamLakRampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * This class implements the 2D kernel of the atract filter.
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class AtractKernel2D extends Grid2DComplex {

	public AtractKernel2D(final int sizeX, final int sizeY) {
		this(sizeX, sizeY, 3);
	}

	public AtractKernel2D(final int sizeX, final int sizeY, final float zeroValue) {
		super(sizeX, sizeY);
		final int paddedSizeRealX = getSize()[0];
		final int paddedSizeRealY = getSize()[1];
		//Configuration.loadConfiguration();
		//final float D = (float) Configuration.getGlobalConfiguration().getGeometry().getSourceToDetectorDistance();
		//final float R = (float) Configuration.getGlobalConfiguration().getGeometry().getSourceToAxisDistance();
		//final float c = (float) (1/(Math.PI*Math.PI*4)*R/D);
		final float c = 1;
		setAtIndex(0, 0, zeroValue);
		for (int i = 0; i < paddedSizeRealX / 2; i++) {
			int i2 = i * i;
			for (int j = 0; j < paddedSizeRealY / 2; j++) {
				if (!(i == 0 && j == 0))
					setAtIndex(i, j, c * Math.abs(j) / (i2 + j * j));
			}
		}
		for (int i = paddedSizeRealX / 2; i < paddedSizeRealX; i++) {
			int tmpI = paddedSizeRealX - i;
			int tmpI2 = tmpI * tmpI;
			for (int j = 0; j < paddedSizeRealY / 2; j++) {
				setAtIndex(i, j, c * Math.abs(j) / (tmpI2 + j * j));
			}
		}
		for (int j = paddedSizeRealY / 2; j < paddedSizeRealY; j++) {
			int tmpJ = paddedSizeRealY - j;
			int tmpJ2 = tmpJ * tmpJ;
			for (int i = 0; i < paddedSizeRealX / 2; i++) {
				setAtIndex(i, j, c * Math.abs(tmpJ) / (i * i + tmpJ2));
			}
		}
		for (int i = paddedSizeRealX / 2; i < paddedSizeRealX; i++) {
			int tmpI = paddedSizeRealX - i;
			int tmpI2 = tmpI * tmpI;
			for (int j = paddedSizeRealY / 2; j < paddedSizeRealY; j++) {
				int tmpJ = paddedSizeRealY - j;
				setAtIndex(i, j, c * Math.abs(tmpJ) / (tmpI2 + tmpJ * tmpJ));
			}
		}

		transformForward();
		//	getRealSubGrid(0, 0, paddedSizeRealX, paddedSizeRealY).show();
	}

	public void applyToGrid(Grid2D input) {

		Grid2DComplex subGrid = new Grid2DComplex(input);

		subGrid.transformForward();
		for (int i = 0; i < subGrid.getSize()[0]; ++i) {
			for (int j = 0; j < subGrid.getSize()[1]; j++) {
				subGrid.multiplyAtIndex(i, j, getRealAtIndex(i, j), getImagAtIndex(i, j));
			}
		}
		subGrid.transformInverse();

		Grid2D filteredSinoSub = subGrid.getRealSubGrid(0, 0, input.getSize()[0], input.getSize()[1]);
		for (int i = 0; i < filteredSinoSub.getSize()[0]; i++) {
			for (int j = 0; j < filteredSinoSub.getSize()[1]; j++) {
				input.setAtIndex(i, j, filteredSinoSub.getAtIndex(i, j));
			}
		}
	}

	public void applyToGrid(Grid3D input) {

		int iter = input.getSize()[2];

		for (int i = 0; i < iter; i++) {
			applyToGrid(input.getSubGrid(i));
		}
	}

	public final static void main(String[] args) {
		new ImageJ();

		int size1 = 1240;
		int size2 = 960;
		// 2D example
		Grid2D in2D = new Grid2D(size1, size2);
		for (int j = 0; j < in2D.getHeight(); j++) {
			for (int i = 0; i < in2D.getWidth(); i++) {
				if (Math.sqrt(Math.pow((i - size1 / 2), 2) + Math.pow((j - size2 / 2), 2)) < 0.25
						* Math.min(size1, size2))
					in2D.setAtIndex(i, j, 2);
			}
		}
		Grid2D in = new Grid2D(in2D);

		AtractKernel2D r = new AtractKernel2D(size1, size2);
		r.show("Atract Residual Kernel 2D");
		LaplaceKernel2D kernel = new LaplaceKernel2D();
		in2D.show("Before");
		kernel.applyToGrid(in2D);
		in2D.show("After Laplace");
		r.applyToGrid(in2D);
		in2D.show("After Residual");

		RampFilter rf = new RamLakRampFilter();
		try {
			rf.configure();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		RampFilteringTool rft = new RampFilteringTool();
		rft.setRamp(rf);
		rft.setConfigured(true);

		rft.applyToolToImage(in).show();

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
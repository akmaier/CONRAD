package edu.stanford.rsl.tutorial.atract;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;

/**
 * This class combines the LaplacianKernel2D, the BorderRemoval2D and the residual AtractKernel
 * in order to enable reconstruction on truncated data
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class AtractFilter2D {

	/**
	 * This method implements the AtractFilter using the 2-D Implementation of the AtractKernel
	 * @param input Projection stack
	 */
	public void applyToGrid2D(Grid3D input) {
		int maxU = input.getSize()[0];
		int maxV = input.getSize()[1];
		LaplaceKernel2D lap = new LaplaceKernel2D();
		lap.applyToGrid(input);
		input.show("Laplace");
		BorderRemoval2D bord = new BorderRemoval2D();
		bord.applyToGrid(input);
		input.show("BorderRemoval");
		AtractKernel2D atract = new AtractKernel2D(maxU, maxV);
		atract.applyToGrid(input);
		input.show("Atract");

	}

	/**
	 * This method implements the AtractFilter using the 2-D Implementation of the AtractKernel
	 * @param input Projection stack
	 */
	public void applyToGrid2D(Grid2D input) {
		int maxU = input.getSize()[0];
		int maxV = input.getSize()[1];
		LaplaceKernel2D lap = new LaplaceKernel2D();
		lap.applyToGrid(input);
		input.show("Laplace");
		BorderRemoval1D bord = new BorderRemoval1D();
		bord.applyToGridLap2D(input);
		input.show("BorderRemoval");
		AtractKernel2D atract = new AtractKernel2D(maxU, maxV);
		atract.applyToGrid(input);
		input.show("Atract");

	}

	/**
	 * This method implements the AtractFilter using the 1-D Implementation of the AtractKernel
	 * @param input Projection stack
	 */
	public void applyToGrid1D(Grid3D input) {
		int maxU = input.getSize()[0];

		LaplaceKernel1D lap = new LaplaceKernel1D(maxU);
		lap.applyToGrid(input);

		BorderRemoval1D bord = new BorderRemoval1D();
		bord.applyToGridLap1D(input);

		AtractKernel1D atract = new AtractKernel1D(maxU);
		atract.applyToGrid(input);

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
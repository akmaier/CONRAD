/*
 * Copyright (C) 2014 - Anja Pohan and Stefan Nottrott 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.opencl;

import ij.ImageJ;


import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

public class OpenCLSynchProblemExample {
	public static void main(String[] args){
		Phantom grid = new UniformCircleGrid2D(512,512);
		OpenCLGrid2D grid1 = new OpenCLGrid2D(grid);
		float val = 1.f;
		// This is a device operation
		NumericPointwiseOperators.addBy(grid1, val);
		// This is a host operation (because getSubGrid returns a grid that is no OpenCL grid!)
		NumericPointwiseOperators.addBy(grid1.getSubGrid(256),8);
		//grid1.notifyAfterWrite();
		// This is a device operation again BUT: The previous host operation was executed on a subgrid --> No host update flag was set in the 2D grid!!
		// Thus, the previous host change will just be ignored and the "old" memory state from the GPU will be used
		NumericPointwiseOperators.addBy(grid1, val);
		//grid1.deactivateCL();
		new ImageJ();
		grid1.show();
	}
}
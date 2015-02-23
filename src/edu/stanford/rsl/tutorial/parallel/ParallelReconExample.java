package edu.stanford.rsl.tutorial.parallel;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import ij.ImageJ;

/**
 * This is a simple example of a parallel-beam reconstruction.
 * First, a phantom is created. Then we generate parallel projections using a parallel projector.
 * This allows us to visualize the sinogram.
 * For reconstruction, we filter the sinogram for every detector row.
 * Then, the sinogram is backprojected.
 * 
 * @author Shiyang Hu
 *
 */
public class ParallelReconExample {

	public static void main (String [] args){
		new ImageJ();
		
		int x = 200;
		int y = 200;
		// Create a phantom
		Phantom phan = new DotsGrid2D(x, y);
		//phan = new UniformCircleGrid2D(x, y);
		//phan = new MickeyMouseGrid2D(x, y);
		phan.show("The Phantom");
		
		// Project forward parallel
		ParallelProjector2D projector = new ParallelProjector2D(Math.PI, Math.PI/180.0, 400, 1);
		Grid2D sinogram = projector.projectRayDrivenCL(phan);
		sinogram.show("The Sinogram");
		Grid2D filteredSinogram = new Grid2D(sinogram);
		
		
		// Filter with RamLak
		RamLakKernel ramLak = new RamLakKernel(400, 1);
		for (int theta = 0; theta < sinogram.getSize()[1]; ++theta) {
			ramLak.applyToGrid(filteredSinogram.getSubGrid(theta));
		}
		filteredSinogram.show("The Filtered Sinogram");
		
		// Backproject and show
		ParallelBackprojector2D backproj = new ParallelBackprojector2D(200, 200, 1, 1);
		backproj.backprojectPixelDriven(filteredSinogram).show("The Reconstruction");
		
	}
	
}
/*
 * Copyright (C) 2010-2014 Shiyang Hu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
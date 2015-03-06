package edu.stanford.rsl.tutorial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

public class FanBeamParallelBeamComparison {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		
		// sinogram params
		double focalLength = 800, 
				maxT = 300,
				deltaT = 1.0,
				gammaM = Math.atan((maxT / 2.f - 0.5) / focalLength), 
				maxBeta = Math.PI + 2*gammaM, 
				deltaBeta = maxBeta / 180;
		// image params
		int imgSzXMM = 200,            // [mm]
			imgSzYMM = imgSzXMM;       // [mm]
		float pxSzXMM = 1.0f,          // [mm]
			pxSzYMM = pxSzXMM;         // [mm]

		//float focalLength = 400, maxBeta = (float) Math.PI*2, deltaBeta = maxBeta / 200, maxT = 200, deltaT = 1;

		int phantomType = 2; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
							// 3=DotsGrid
		// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
			imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();

		// image object
		Phantom phantom;
		switch (phantomType) {
		case 0:
			phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
			break;
		case 1:
			phantom = new MickeyMouseGrid2D(imgSzXGU, imgSzYGU);
			break;
		case 2:
			phantom = new TestObject1(imgSzXGU, imgSzYGU);
			break;
		case 3:
			phantom = new DotsGrid2D(imgSzXGU, imgSzYGU);
			break;
		default:
			phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
			break;
		}

		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2.0,
				-(imgSzYGU * phantom.getSpacing()[1]) / 2.0);
		phantom.show();
		Grid2D grid = phantom;


		
		// Fan Beam Projection
		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
		Grid2D fanBeamSinoRay = fanBeamProjector.projectRayDriven(grid);
		fanBeamSinoRay.show("FB CL Sinogram");
		
		// Parallel for comparison
		ParallelProjector2D Projector = new ParallelProjector2D(maxBeta, deltaBeta, maxT, deltaT);
		Grid2D pBeamSinoRay = Projector.projectRayDriven(grid);
		pBeamSinoRay.show("FB CL Sinogram");
		
		
		// Parker Weights
		ParkerWeights pWeights = new ParkerWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
		pWeights.show();
		pWeights.applyToGrid(fanBeamSinoRay);
		fanBeamSinoRay.show("Fan Beam After Parker Weights");
		
		// Cosine Weights
		CosineFilter cosFilt = new CosineFilter(focalLength, maxT, deltaT);
		
		
		// Filtering
		RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
		for (int theta = 0; theta < fanBeamSinoRay.getSize()[0]; ++theta) {
			cosFilt.applyToGrid(fanBeamSinoRay.getSubGrid(theta));
			ramLak.applyToGrid(fanBeamSinoRay.getSubGrid(theta));
			ramLak.applyToGrid(pBeamSinoRay.getSubGrid(theta));
		}
		fanBeamSinoRay.show("Fan Beam Filtered Sinogram");
		pBeamSinoRay.show("Parallel Beam Filtered Sinogram");

		// Backproject fan beam
		FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSzXMM, imgSzYMM);
		Grid2D fanbeamResult = fbp.backprojectPixelDriven(fanBeamSinoRay);
		fanbeamResult.show("Fan Beam Result");
		
		// Backproject parallel beam
		ParallelBackprojector2D pbp = new ParallelBackprojector2D(imgSzXMM,imgSzYMM,pxSzXMM,pxSzYMM);
		Grid2D parallelResult = pbp.backprojectPixelDriven(pBeamSinoRay);
		parallelResult.show("Parallel Beam Result");
	
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
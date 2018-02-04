package edu.stanford.rsl.tutorial.atract;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

public class AtractExample {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// sinogram params
		double maxTheta = Math.PI, // [rad]
				deltaTheta = Math.PI / 360, // [rad]
				maxS = 300, // [mm]
				deltaS = 1.0; // [mm]
		// image params
		int imgSzXMM = 200, // [mm]
				imgSzYMM = imgSzXMM; // [mm]
		float pxSzXMM = 1.0f, // [mm]
				pxSzYMM = pxSzXMM; // [mm]

		int maxSIndex = (int) (maxS / deltaS + 1);
		int maxThetaIndex = (int) (maxTheta / deltaTheta + 1);

		//float focalLength = 400, maxBeta = (float) Math.PI*2, deltaBeta = maxBeta / 200, maxT = 200, deltaT = 1;

		int phantomType = 1; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
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
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2.0, -(imgSzYGU * phantom.getSpacing()[1]) / 2.0);
		phantom.show();
		Grid2D grid = phantom;

		// create projections
		ParallelProjector2D projector = new ParallelProjector2D(maxTheta, deltaTheta, maxS, deltaS);
		Grid2D sinoRay = projector.projectRayDriven(grid);
		sinoRay.show("Sinogram");

		// Apply constant kollimator here
		Kollimator koll = new Kollimator(maxThetaIndex, maxSIndex);
		koll.applyToGrid(sinoRay, 100);
		Grid2D sinoWithoutATRACT = new Grid2D(sinoRay);
		sinoRay.show("Sinogram after Kollimation");

		// Apply ATRACT filter
		AtractFilter1D at = new AtractFilter1D();
		at.applyToGrid(sinoRay);

		// Apply RamLak filter
		RamLakKernel ramLak = new RamLakKernel((int) (maxS / deltaS), deltaS);
		for (int theta = 0; theta < sinoWithoutATRACT.getSize()[0]; ++theta) {
			ramLak.applyToGrid(sinoWithoutATRACT.getSubGrid(theta));
		}

		// Backproject with ATRACT
		ParallelBackprojector2D bp = new ParallelBackprojector2D(imgSzXMM, imgSzYMM, pxSzXMM, pxSzYMM);
		Grid2D reconWithATRACT = bp.backprojectRayDriven(sinoRay);
		reconWithATRACT.show("Reconstruction with ATRACT");

		// Backproject without ATRACT
		Grid2D reconWithoutATRACT = bp.backprojectRayDriven(sinoWithoutATRACT);
		reconWithoutATRACT.show("Reconstruction with RamLak");

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
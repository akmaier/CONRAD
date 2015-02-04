package edu.stanford.rsl.tutorial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

/**
 * Simple example that computes and displays two forward projections of a uniform circle.
 * @author Recopra Seminar Summer 2012
 *
 */
public class DisplaySinogram {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// sinogram params
		double maxTheta=Math.PI,       // [rad]
			   deltaTheta=Math.PI/200, // [rad]
			   maxS=150,               // [mm]
			   deltaS=1.0;             // [mm]
		// image params
		int imgSzXMM = 150,            // [mm]
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

		ParallelProjector2D projector = new ParallelProjector2D(maxTheta, deltaTheta, maxS, deltaS);

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
		
		// create projections
		Grid2D sinoRayCL = projector.projectRayDrivenCL(grid);
		sinoRayCL.show("Sino CL Ray");
		Grid2D sinoRay = projector.projectRayDriven(grid);
		sinoRay.show("Sino CPU Ray");
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
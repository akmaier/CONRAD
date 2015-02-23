package edu.stanford.rsl.tutorial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.filters.RayByRayFiltering;
import edu.stanford.rsl.tutorial.phantoms.Ellipsoid;
import edu.stanford.rsl.tutorial.phantoms.Phantom;

public class RayByRayFilterExample{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// image params
		int imgSzXMM = 256, // [mm]
				imgSzYMM = imgSzXMM; // [mm]
		float pxSzXMM = 1.0f, // [mm]
				pxSzYMM = pxSzXMM; // [mm]
		// fan beam bp parameters
		double gammaM = 11.768288932020647*Math.PI/180, 
				maxT = 500, 
				deltaT = 1.0, 
				focalLength = (maxT/2.0-0.5)*deltaT/Math.tan(gammaM),
				maxBeta = 285.95*Math.PI/180,//+gammaM*2, 
				deltaBeta = maxBeta / 132;

		System.out.println(gammaM*180/Math.PI);

		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
				imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();

		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(
				focalLength, maxBeta, deltaBeta, maxT, deltaT);

		Phantom phantom = new Ellipsoid(imgSzXGU, imgSzYGU);

		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2, -(imgSzYGU * phantom.getSpacing()[1]) / 2);
		//phantom.setOrigin(-50.0, -50.0);

		phantom.show();

		Grid2D projectionP = new Grid2D(phantom);

		for (int iter =0; iter < 1; iter ++) {
			// create projections
			Grid2D fanBeamSinoRay = fanBeamProjector.projectRayDriven(projectionP);
			fanBeamSinoRay.clone().show("Sinogram");

			// because the log of 0 is -inf, we need to scale down
			NumericPointwiseOperators.divideBy(fanBeamSinoRay, 40);

			//apply noise
			LogPoissonNoiseFilteringTool pnoise = new LogPoissonNoiseFilteringTool();

			try {
				fanBeamSinoRay = pnoise.applyToolToImage(fanBeamSinoRay);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// scale up again
			NumericPointwiseOperators.multiplyBy(fanBeamSinoRay, 40);

			fanBeamSinoRay.clone().show("SinogramNoised");

			int weightType = 0;

			Grid2D RedundancyWeights = new ParkerWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);

			RedundancyWeights.show("Current Weight");

			NumericPointwiseOperators.multiplyBy(fanBeamSinoRay, RedundancyWeights);

			// Create 11 filters, set their parameters
			RayByRayFiltering rbrF = new RayByRayFiltering((int) (maxT / deltaT), deltaT, 0.5, 
					0.000200, 1., 1000000, 6, 405);
			
			// Show the filters
			rbrF.showFilters();
			
			// Apply the filters
			for (int i =0; i < fanBeamSinoRay.getSize()[1] ;  i ++) {  
				rbrF.applyToGrid(fanBeamSinoRay.getSubGrid(i));
			}

			fanBeamSinoRay.show("After Filtering Ray by Ray");

			// Do the backprojection
			FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength,
					deltaT, deltaBeta, imgSzXMM, imgSzYMM);

			Grid2D reco = fbp.backprojectPixelDriven(fanBeamSinoRay);
			reco.show("Reco Ray by Ray" + iter);

			NumericGrid recoDiff = NumericPointwiseOperators.subtractedBy(phantom, reco);
			recoDiff.show("RecoDiff Ray by Ray" + iter);

		}
	}
}
/*
 * Copyright (C) 2010-2014 Salaheldin Saleh, me@s-saleh.com
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
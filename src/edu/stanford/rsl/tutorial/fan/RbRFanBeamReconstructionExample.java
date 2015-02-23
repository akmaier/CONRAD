package edu.stanford.rsl.tutorial.fan;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.BinaryWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.CompensationWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.SilverWeights;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.filters.RayByRayFiltering;
import edu.stanford.rsl.tutorial.phantoms.Ellipsoid;
import edu.stanford.rsl.tutorial.phantoms.Phantom;

public class RbRFanBeamReconstructionExample{


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

		int phantomType = 4; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
		// 3=DotsGrid, 4=ellipsoid
		// size in grid units
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

			Grid2D RedundancyWeights;
			switch (weightType) {
			case 0:
				RedundancyWeights = new ParkerWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
				break;
			case 1:
				RedundancyWeights = new SilverWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
				break;
			case 2:
				RedundancyWeights = new CompensationWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
				break;
			case 3:
				RedundancyWeights = new BinaryWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
				break;
			default:
				RedundancyWeights = new CompensationWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
			}

			RedundancyWeights.show("Current Weight");

			NumericPointwiseOperators.multiplyBy(fanBeamSinoRay, RedundancyWeights);

			// Filter backprojection with ramp filter only
			Grid2D fanBeamSinoRayOriginal = new Grid2D(fanBeamSinoRay);
			// A non filtered sinogram version
			Grid2D fanBeamSinoRayNoFilter = new Grid2D(fanBeamSinoRay);

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

			////////////////////////

			// Ramp filter only
			FanBeamBackprojector2D fbp2 = new FanBeamBackprojector2D(focalLength,
					deltaT, deltaBeta, imgSzXMM, imgSzYMM);
			RamLakKernel ramLak2 = new RamLakKernel((int) (maxT / deltaT), deltaT);
			for (int theta = 0; theta < fanBeamSinoRayOriginal.getSize()[1]; ++theta) {
				ramLak2.applyToGrid(fanBeamSinoRayOriginal.getSubGrid(theta));
			}
			fanBeamSinoRayOriginal.show("After Filtering Ramp Only");
			Grid2D reco2 = fbp2.backprojectPixelDriven(fanBeamSinoRayOriginal);
			reco2.show("Reco Ramp Only" + iter);
			NumericGrid recoDiff2 = NumericPointwiseOperators.subtractedBy(phantom, reco2);
			recoDiff2.show("RecoDiff Ramp Only" + iter);

			////////////////////////

			// No filters
			FanBeamBackprojector2D fbp3 = new FanBeamBackprojector2D(focalLength,
					deltaT, deltaBeta, imgSzXMM, imgSzYMM);

			Grid2D reco3 = fbp3.backprojectPixelDriven(fanBeamSinoRayNoFilter);
			reco3.show("Reco NoFilter" + iter);
			NumericGrid recoDiff3 = NumericPointwiseOperators.subtractedBy(phantom, reco);
			recoDiff3.show("RecoDiff NoFilter" + iter);


		}
	}

}
/*
 * Copyright (C) 2010-2014 Salaheldin Saleh, me@s-saleh.com
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
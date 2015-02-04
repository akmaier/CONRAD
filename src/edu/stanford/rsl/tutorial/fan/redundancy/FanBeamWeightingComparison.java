package edu.stanford.rsl.tutorial.fan.redundancy;



import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;



import ij.ImageJ;
import ij.ImagePlus;
import ij.io.Opener;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.RamLakKernel;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.SilverWeights;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

public class FanBeamWeightingComparison {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// image params
		int imgSzXMM = 200,            // [mm]
			imgSzYMM = imgSzXMM;       // [mm]
		float pxSzXMM = 1f,          // [mm]
			pxSzYMM = pxSzXMM;         // [mm]
		
		// sinogram params
		double fanAngle = 10.0*Math.PI/180.0, 
				maxT = 300, // Detector length according to image dimensions to avoid truncation
				deltaT = 1,
				//gammaM = Math.atan((maxT / 2.f - 0.5) / focalLength), 
				maxBeta = 159 * Math.PI /180.0, 
				deltaBeta = maxBeta / 360.0;
		double focalLength = (maxT/2.0-0.5)/Math.tan(fanAngle);


		//float focalLength = 400, maxBeta = (float) Math.PI*2, deltaBeta = maxBeta / 200, maxT = 200, deltaT = 1;

		int phantomType = 6; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
							// 3=DotsGrid
		// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
			imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();

		Opener op = new Opener();
		
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
		case 4:
			phantom = new SheppLogan(imgSzXGU);
			break;
		case 5:
			// Load your own saved image here!!
			ImagePlus ipl = op.openZip("D:\\!Stanford\\Patent\\PaperCase\\Phantom_Paper.zip");
			phantom = new DotsGrid2D(ipl.getWidth(),ipl.getHeight());
			NumericPointwiseOperators.copy(phantom, ImageUtil.wrapImagePlusSlice(ipl, 0, false));
			imgSzXMM = phantom.getWidth();
			imgSzYMM = phantom.getHeight();
			imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM); // [GU]
			imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
			break;
		default:
			phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
			break;
		}

		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2.0,
				-(imgSzYGU * phantom.getSpacing()[1]) / 2.0);
		phantom.show("Phantom");
		Grid2D grid = phantom;


		
		// Fan Beam Projection
		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
		Grid2D Sinogram = fanBeamProjector.projectRayDrivenCL(grid);
		Sinogram.show("Sinogram");
		
		
		// Redundancy Weights
		List<Grid2D> rWeights = new ArrayList<Grid2D>();
		// No weights
		rWeights.add(
			new Grid2D(Sinogram.getSize()[0], Sinogram.getSize()[1])
		);
		// Silver / Parkerlike weights
		rWeights.add(new SilverWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta));
		// Compensation Weights
		rWeights.add(new CompensationWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta));
		
		for (int i=0; i < rWeights.get(0).getSize()[0]; ++i)
		{
			for (int j=0; j < rWeights.get(0).getSize()[1]; ++j)
			{
				rWeights.get(0).setAtIndex(i, j, 1.f);
			}
		}
		
		//Show weights
		for (Iterator<Grid2D> it = rWeights.iterator(); it.hasNext(); )
			it.next().show();
		
		
		// Cosine Weights
		CosineFilter cosFilt = new CosineFilter(focalLength, maxT, deltaT);
		
		// Filtering
		RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
		
		// Backproject fan beam
		FanBeamBackprojector2D backprojector = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSzXMM, imgSzYMM);
		
		for (int i=0; i < rWeights.size(); ++i)
		{
			Grid2D sino = new Grid2D(Sinogram);
			Grid2D result = new Grid2D(FBP(sino, cosFilt, rWeights.get(i), ramLak, backprojector));
			
			switch(i)
			{
			case 0:
				result.show("No Weighting");
				break;
			case 1: 
				result.show("Silver Weighting");
				break;
			case 2:
				result.show("Compensation Weighting");
				break;
			default:
				result.show("Compensation Weighting");
				break;
			}
		}
	
	}
		
	private static Grid2D FBP( Grid2D Sinogram, CosineFilter cosFilt, Grid2D RedundancyWeights, RamLakKernel ramLak, FanBeamBackprojector2D backprojector)
	{
		
		NumericPointwiseOperators.multiplyBy(Sinogram, RedundancyWeights);
		for (int theta = 0; theta < Sinogram.getSize()[1]; ++theta) {
			cosFilt.applyToGrid(Sinogram.getSubGrid(theta));
			ramLak.applyToGrid(Sinogram.getSubGrid(theta));
		}
		
		return backprojector.backprojectPixelDriven(Sinogram);
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
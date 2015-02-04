package edu.stanford.rsl.tutorial.atract;

import ij.ImageJ;
import ij.ImagePlus;
import ij.io.Opener;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.dynamicCollimation.copyRedundantData;
import edu.stanford.rsl.tutorial.fan.redundancy.BinaryWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.RamLakKernel;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.FilePhantom;
import edu.stanford.rsl.tutorial.phantoms.MTFphantom;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

public class AtractWithBinaryWeight {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double speedup = 1;
		
		// image params
		int imgSzXMM = (int)(1024/speedup),            // [mm]
			imgSzYMM = imgSzXMM;       // [mm]
		float pxSzXMM = 1.0f,          // [mm]
			pxSzYMM = pxSzXMM;         // [mm]
		
		// sinogram params
		
		@SuppressWarnings("unused")
		double	focalLength = 4416/speedup,
				maxT = 1472/speedup, // Detector length according to image dimensions to avoid truncation
				deltaT = 1.0,
				//gammaM = Math.atan((maxT / 2.f - 0.5) / focalLength), 
				maxBeta = 200*Math.PI/180, 
				//deltaBeta = maxBeta / 165,
				deltaBeta = maxBeta / 165,
				fanAngle = Math.atan((maxT/2.0-0.5*deltaT)/focalLength);
		
		/*double	fanAngle = 10*Math.PI/180,//Math.atan((maxT/2.0-0.5*deltaT)/focalLength);
				maxT = 1472/speedup, // Detector length according to image dimensions to avoid truncation
				deltaT = 1.0,
				focalLength = (maxT-0.5f*deltaT)*0.5f/Math.tan(fanAngle),//4416/speedup,
				//gammaM = Math.atan((maxT / 2.f - 0.5) / focalLength), 
				maxBeta = 200*Math.PI/180, 
				deltaBeta = maxBeta / 165;*/

		int phantomType = 4; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
							// 3=DotsGrid, 4=SheppLogan, 5=MTF, 6=File
		// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
			imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();

		
		// image object
		Phantom phantom;
		Grid2D Sinogram = null;
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
			phantom = new MTFphantom(imgSzXGU, imgSzYGU, 0.95, 0.7895, 1.f, 4.f);
			break;
		case 6:
			//phantom = new FilePhantom(imgSzXGU, imgSzYGU, "D:\\!Stanford\\RecoSVN\\berger\\Data\\PigSlice.zip");
			//phantom = new FilePhantom(imgSzXGU, imgSzYGU, "C:\\Users\\Martin\\Desktop\\lena.zip");
			phantom = new FilePhantom(imgSzXGU, imgSzYGU, "C:\\Users\\Martin\\Desktop\\donald.zip");
			break;
		case 7:
			// Load your own saved image here!!
			Opener op = new Opener();
			ImagePlus ipl = op.openZip("D:\\!Stanford\\RecoSVN\\berger\\MICCAI\\Results\\MTF\\MTFPhantom_1024_1024_r2_384.zip");
			phantom = new DotsGrid2D(ipl.getWidth(),ipl.getHeight());
			imgSzXMM = phantom.getWidth();
			imgSzYMM = phantom.getHeight();
			imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM); // [GU]
			imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
			
			
			Sinogram = ImageUtil.wrapImagePlusSlice(op.openZip("D:\\!Stanford\\RecoSVN\\berger\\MICCAI\\Results\\MTF\\Sinogram_1024_1024.zip"),0, false);
			Sinogram.setSpacing(deltaBeta, deltaT);
			
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
		
		if (phantomType != 7)
		{
			// Fan Beam Projection
			FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			Sinogram = fanBeamProjector.projectRayDrivenCL(grid);
		}
		Sinogram.show("Sinogram");
		
		
		// ****************** From here on we have a valid sinogram ******************************
		
		
		// Binary Weights
		BinaryWeights binWeights = new BinaryWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
		
		//binWeights.show("Binary Weights");
		
		// Parker Weights
		ParkerWeights parkWeights = new ParkerWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
		//parkWeights.show("Parker Weights");
		
		// apply parker weights
		Grid2D parkerSinogram = new Grid2D(Sinogram);
		NumericPointwiseOperators.multiplyBy(parkerSinogram, parkWeights);
		//parkerSinogram.show("Parker Weights Sinogram");
		
		// apply binary Weights (artificial dynamic collimator)
		Grid2D binSinogram = new Grid2D(Sinogram);
		NumericPointwiseOperators.multiplyBy(binSinogram, binWeights);
		//binSinogram.show("Binary Weights Sinogram");
		
		// try to recover missing (collimated) data by known redundant part and use parker weighting
		copyRedundantData copyClass = new copyRedundantData(focalLength, maxT, deltaT, maxBeta, deltaBeta);
		Grid2D copiedFullSinogram = new Grid2D(binSinogram);
		copyClass.applyToGrid(copiedFullSinogram);
		NumericPointwiseOperators.multiplyBy(copiedFullSinogram, parkWeights);
		// In this case we apply two weighting schemes: binary weights and parker weights!
		// therefore the scale correction was done twice, so we need to divide once by the same scaling factor
		NumericPointwiseOperators.multiplyBy(copiedFullSinogram, (float)( (Math.PI) / maxBeta));
		//copiedFullSinogram.show("Copied Full Sinogram");
		
		// show difference between parker weighting on full data and parker on copied redundancies
		Grid2D nonColl_park_Sino_minus_Coll_copied_park_Sino = (Grid2D)NumericPointwiseOperators.subtractedBy(parkerSinogram, copiedFullSinogram);
		nonColl_park_Sino_minus_Coll_copied_park_Sino.show("NonColl_Park_Sino minus Coll_Copied_Park_Sino");
		
		// Cosine Filter
		CosineFilter cosFilt = new CosineFilter(focalLength, maxT, deltaT);
		// Filtering
		RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
		// Create ATRACT filter
		AtractFilter1D atractFilter = new AtractFilter1D();
		
		// Filtering for Binary ATRACT
		Grid2D atractSinogram = new Grid2D(binSinogram);
		Grid2D parkerAtractSinogram = new Grid2D(binSinogram);
		//Grid2D atract2DSinogram = new Grid2D(binSinogram);
		for (int theta = 0; theta < Sinogram.getSize()[0]; ++theta) {
			cosFilt.applyToGrid(atractSinogram.getSubGrid(theta));
			cosFilt.applyToGrid(parkerAtractSinogram.getSubGrid(theta));
			//cosFilt.applyToGrid(atract2DSinogram.getSubGrid(theta));
		}
		
		// apply the Atract kernel
		atractFilter.applyToGrid(atractSinogram, binWeights.getBinaryMask());
		boolean checkMask = false;
		if (checkMask) {
			Grid2D mask = new Grid2D(
				new float[parkerAtractSinogram.getSize()[0]*parkerAtractSinogram.getSize()[1]],
				parkerAtractSinogram.getSize()[0],
				parkerAtractSinogram.getSize()[1]
			);
			NumericPointwiseOperators.fill(mask, 1.0f);
			atractFilter.applyToGrid(parkerAtractSinogram, mask);
		}
		
		/*
		// apply 2D atract
		LaplaceKernel2D laplace2D = new LaplaceKernel2D();
		laplace2D.applyToGrid(atract2DSinogram);
		AtractKernel2D atract2D = new AtractKernel2D(atract2DSinogram.getSize()[0], atract2DSinogram.getSize()[1]);
		atract2D.applyToGrid(atract2DSinogram);
		*/
		
		
		// Filtering for Binary and Parker FBP
		for (int theta = 0; theta < Sinogram.getSize()[0]; ++theta) {
			cosFilt.applyToGrid(binSinogram.getSubGrid(theta));
			cosFilt.applyToGrid(parkerSinogram.getSubGrid(theta));
			cosFilt.applyToGrid(copiedFullSinogram.getSubGrid(theta));
			ramLak.applyToGrid(binSinogram.getSubGrid(theta));
			ramLak.applyToGrid(parkerSinogram.getSubGrid(theta));
			ramLak.applyToGrid(copiedFullSinogram.getSubGrid(theta));
		}
		
		
		/*
		FanBeamBackprojector2D fbp_beads = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, 64, 64);
		
		Grid2D[] beadSinogram = {parkerSinogram, copiedFullSinogram, atractSinogram};
		String[] sinoNames = {"Parker_RamLak_CentralBead","Binary_Copied_Parker_RamLak_CentralBead","Binary_ATRACT_CentralBead"};
		for (int i=0; i<beadSinogram.length; ++i)
		{
			Grid2D parkerResBeads = fbp_beads.backprojectPixelDriven_HighResRegion(beadSinogram[i], 1/16.f, new double[] {0, 0} );
			parkerResBeads.show(sinoNames[i]);
		}
		*/
		
		// Do the backprojections
		
		FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSzXMM, imgSzYMM);
		
		Grid2D parkerRes = fbp.backprojectPixelDrivenCL(parkerSinogram);
		parkerRes.show("Parker_RamLak");
		
		Grid2D binaryRes = fbp.backprojectPixelDrivenCL(binSinogram);
		binaryRes.show("Binary_RamLak");
		
		Grid2D copiedParkerRes = fbp.backprojectPixelDrivenCL(copiedFullSinogram);
		copiedParkerRes.show("Binary_Copied_Parker_RamLak");
		
		Grid2D binaryAtractRes = fbp.backprojectPixelDrivenCL(atractSinogram);
		binaryAtractRes.show("Binary_ATRACT");
		
		//Grid2D Atract2DRes = fbp.backprojectPixelDrivenCL(atract2DSinogram);
		//Atract2DRes.show("2D_ATRACT");
		 
		
		if (checkMask) {Grid2D parkerAtractRes = fbp.backprojectPixelDrivenCL(parkerAtractSinogram);
			parkerAtractRes.show("Parker_ATRACT");
		}
	}

}
/*
 * Copyright (C) 2010-2014  Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
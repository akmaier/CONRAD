/*
 * Copyright (C) 2014 Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.fourierConsistency.wedgefilter;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.BinaryWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.CompensationWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.SilverWeights;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.interpolation.FanParameters;
import edu.stanford.rsl.tutorial.interpolation.Interpolation;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.PohlmannPhantom;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;


/**
 * Example to evaluate the Double Wedge Filter projection estimation method (WF).
 * The Double Wedge filtered reconstruction is compared to three other estimation
 * methods, Linear Interpolation, Spectral Deconvolution (SD) and Iterative 
 * Reconstruction Reprojection (IRR). This example shows the results of the 
 * reconstruction after the projection estimation and computes the L2-Norm.
 *  
 * @author Marcel Pohlmann
 * 
 */
public class ProjectionEstimationExample {


	/** The control parameters are already set by default such that this class recreates 
	 *  the 134 projection result introduced in the CT-Meeting paper:
	 *  "Estimation of Missing Fan-Beam Projections using Frequency Consistency Conditions",
	 *  Pohlmann M., Berger M., Maier A., Hornegger J., Fahrig R. 
	 *  
	 * @param args
	 */
	public static void main(String[] args) {
		// image params
		int 	imgSzXMM = 512, 	// [mm]
				imgSzYMM = imgSzXMM;// [mm]
		float 	pxSzXMM = 1.0f, 	// [mm]
				pxSzYMM = pxSzXMM; 	// [mm]
		
		// fan beam bp parameters
		double 	gammaM = 11.768288932020647*Math.PI/180, 
				maxT = 500,
				deltaT = 1.0, 
				focalLength = (maxT/2.0-0.5)*deltaT/Math.tan(gammaM),
				// we need projections from a full-scan trajectory
				maxBeta = 360*Math.PI/180,
				deltaBeta = maxBeta / 268;
				
		System.out.println("gammaM = " + gammaM*180/Math.PI + "(degrees)");
		System.out.println("focalLength = " + focalLength);
		
		double 	rp = 200.0,
				_L = focalLength/2,
				_D = focalLength/2;
		
		// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
		imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();

		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(
				focalLength, maxBeta, deltaBeta, maxT, deltaT);

		Grid2D phantom = null;
		
		// switch between different phantom types
		// 0 = circle, 1 = MickeyMouse, 2 = TestObject1, 
		// 3 = DotsGrid, 4 = circle of beads, 5 = load custom phantom 
		int phantomType = 4;
		
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
			phantom = new PohlmannPhantom(imgSzXGU, imgSzYGU, 20, rp, true);
			break;
		case 5:
			try {
				String filenameString = FileUtil.myFileChoose("*ima", false);
				ImagePlus imp = IJ.openImage(filenameString);
				phantom = edu.stanford.rsl.conrad.utils.ImageUtil.wrapImagePlusSlice(imp, 1, true);
			} catch (Exception e) {
				e.printStackTrace();
			}
			break;
		default:
			phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
			break;
		}
		
		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2, -(imgSzYGU * phantom.getSpacing()[1]) / 2);
		
		phantom.show();
		Grid2D projectionP = new Grid2D(phantom);
		
		
		for (int iter = 0; iter < 1; iter ++) {
			// create sinogram holding all projections (f)
			Grid2D fanBeamSinoRay = fanBeamProjector.projectRayDrivenCL(projectionP);
			fanBeamSinoRay.clone().show("Sinogram");
			
			// build missing projection mask (w), where
			// 0:=missing projection, 1:=measured projection
			Grid2D w = new Grid2D(fanBeamSinoRay);
			
			for(int i = 0; i < w.getHeight(); i ++) {
				for(int j = 0; j < w.getWidth(); ++j) {
					w.putPixelValue(j, i, 1.0);
				}
			}
			
			int n = 2;
			
			for(int i = 1; i < w.getHeight() - 1 ; i += n) {
				for(int j = 0; j < w.getWidth(); ++j) {
					w.putPixelValue(j, i, 0.0);
				}
			}
			
			// compute under-sampled sinogram (g) by a pixel-wise multiplication of ideal
			// sinogram with the missing projection mask (g = f.*w)
			Grid2D corSino = (Grid2D) NumericPointwiseOperators.multipliedBy(fanBeamSinoRay, w);

			Grid2D sparseSino = (Grid2D) corSino.clone();
			
			// show under-sampled sinogram
			corSino.clone().show("Sparse Sinogram");
			
			
			Grid2D linearInterp = (Grid2D) corSino.clone();
			
			// compute the linear interpolated sinogram
			for(int i = 0; i < linearInterp.getHeight(); i++) {
				for(int j = 0; j < linearInterp.getWidth(); ++j) {
					if(w.getAtIndex(j, i) == 0.0){
						double value = linearInterp.getPixelValue(j, i-1)+linearInterp.getPixelValue(j, i+1);
						linearInterp.putPixelValue(j, i, value/2);
					}
				}
			}
			
			// show linear interpolation result
			linearInterp.clone().show("Linear Interpolated Sinogram");
			
			// initialize missing projections with the mean value of measured projections
			// to compensate the global intensity loss caused by the Double Wedge filter method
			double val = NumericPointwiseOperators.mean(fanBeamSinoRay);
			
			for(int i = 0; i < corSino.getHeight(); i++) {
				for(int j = 0; j < corSino.getWidth(); ++j) {
					if(w.getAtIndex(j, i) == 0.0){
						corSino.putPixelValue(j, i, val);
					}
				}
			}
			
			// show the initialized under-sampled sinogram
			corSino.clone().show("Sinogram with initial values");
			
			// define parameter array:=[r_p, _L, _D] for the iterativeWedgeFilter(..) method
			double[] parameters = {rp, _L, _D};
			
			// define fan beam parameter container for the IRR(..) method
			FanParameters params = new FanParameters(new double[]{gammaM, maxT, deltaT, focalLength, maxBeta, deltaBeta});
			
			// define maximum iteration values for 
			int 	maxIter_IRR = 3,
					maxIter_SD = 100,
					maxIter_WF = 50;
			
			// switch between patch-wise and global SD 
			boolean patchwiseSD = false;
			
			Grid2D sparseSino_cp = (Grid2D) sparseSino.clone();
			
			// compute missing projections using IRR
			System.out.println("Running Iterative Repropection Reconstruction (IRR) ... ");
			long time_IRR_start = System.currentTimeMillis();
			Grid2D filteredSinogram_IRR = Interpolation.IRR((Grid2D) sparseSino.clone(), w, params, maxIter_IRR, true, new int[]{imgSzXMM, imgSzYMM}, new float[]{pxSzXMM, pxSzYMM});
			long time_IRR_end = System.currentTimeMillis();
			
			// compute missing projections using WF
			System.out.println("Running Wedge Filter (WF) ... ");
			long time_WF_start = System.currentTimeMillis();
			Grid2D filteredSinogram_WF = Interpolation.iterativeWedgeFilter((Grid2D) corSino.clone(), w, maxIter_WF, true, parameters);
			long time_WF_end = System.currentTimeMillis();
			
			// compute missing projections using SD
			System.out.println("Running Spectral Deconvolution (SD) ... ");
			long time_SD_start;
			Grid2D filteredSinogram_SD;
			if(patchwiseSD) {
				int[] patch_size = {64, 64};
				time_SD_start = System.currentTimeMillis();
				filteredSinogram_SD = Interpolation.ApplyPatchwiseSpectralDeconvolution(sparseSino_cp, w, patch_size, maxIter_SD, true);
			} else {
				time_SD_start = System.currentTimeMillis();
				filteredSinogram_SD = Interpolation.SpectralDeconvoltion(sparseSino_cp, w, maxIter_SD, true);
			}
			long time_SD_end = System.currentTimeMillis();
			
			// Print runtime for different Interpolation methods
			System.out.println("Runtime IRR: " + (time_IRR_end - time_IRR_start));
			System.out.println("Runtime WF: " + (time_WF_end - time_WF_start));
			System.out.println("Runtime SD: " + (time_SD_end - time_SD_start));
			

			filteredSinogram_IRR.clone().show("Inpainted Forward Projections");
			filteredSinogram_WF.clone().show("Wedge Filtered Sinogram");
			filteredSinogram_SD.clone().show("Spec Deconv");
			
			double[] spatialSpacing = fanBeamSinoRay.getSpacing();
			
			// make a new copy of IRR result in order to fix problems with the OpenCL backprojector
			filteredSinogram_IRR = (Grid2D) filteredSinogram_IRR.clone();
			
			// spacing needs to be set otherwise backprojector will not work
			linearInterp.setSpacing(spatialSpacing);
			filteredSinogram_IRR.setSpacing(spatialSpacing);
			filteredSinogram_WF.setSpacing(spatialSpacing);
			
			// switch between different redundancy weights
			// 0 := ParkerWeights, 1 := SilverWeights
			// 2 := CompensationWeights, 3 := BinaryWeights
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
			
			// apply redundancy weights
			NumericPointwiseOperators.multiplyBy(fanBeamSinoRay, RedundancyWeights);
			NumericPointwiseOperators.multiplyBy(linearInterp, RedundancyWeights);
			NumericPointwiseOperators.multiplyBy(filteredSinogram_IRR, RedundancyWeights);
			NumericPointwiseOperators.multiplyBy(filteredSinogram_WF, RedundancyWeights);
			NumericPointwiseOperators.multiplyBy(filteredSinogram_SD, RedundancyWeights);
			
			RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
			CosineFilter cKern = new CosineFilter(focalLength, maxT, deltaT);
			
			// apply RamLak and Cosine-filter
			for (int theta = 0; theta < fanBeamSinoRay.getSize()[1]; ++theta) {
				// Cosine kernel
				cKern.applyToGrid(fanBeamSinoRay.getSubGrid(theta));
				cKern.applyToGrid(linearInterp.getSubGrid(theta));
				cKern.applyToGrid(filteredSinogram_IRR.getSubGrid(theta));
				cKern.applyToGrid(filteredSinogram_WF.getSubGrid(theta));
				cKern.applyToGrid(filteredSinogram_SD.getSubGrid(theta));
				
				// RamLak kernel
				ramLak.applyToGrid(fanBeamSinoRay.getSubGrid(theta));
				ramLak.applyToGrid(linearInterp.getSubGrid(theta));
				ramLak.applyToGrid(filteredSinogram_IRR.getSubGrid(theta));
				ramLak.applyToGrid(filteredSinogram_WF.getSubGrid(theta));
				ramLak.applyToGrid(filteredSinogram_SD.getSubGrid(theta));
			}
			
			// make instance of backprojector
			FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength,
					deltaT, deltaBeta, imgSzXMM, imgSzYMM);
			
			// do the backprojection
			Grid2D reco = fbp.backprojectPixelDrivenCL(fanBeamSinoRay);
			Grid2D recoLin = fbp.backprojectPixelDrivenCL(linearInterp);
			Grid2D recoIRR = fbp.backprojectPixelDrivenCL(filteredSinogram_IRR);
			Grid2D recoWF = fbp.backprojectPixelDrivenCL(filteredSinogram_WF);
			Grid2D recoSD = fbp.backprojectPixelDrivenCL(filteredSinogram_SD);
			
			// set all negative values to zero
			NumericPointwiseOperators.removeNegative(reco);
			NumericPointwiseOperators.removeNegative(recoLin);
			NumericPointwiseOperators.removeNegative(recoIRR);
			NumericPointwiseOperators.removeNegative(recoWF);
			NumericPointwiseOperators.removeNegative(recoSD);
			
			// set values that are outside the detector-FOV to zero
			setOutsideFOV(reco, (float) (maxT/2 - 0.1*maxT/2), 0.0f);
			setOutsideFOV(recoLin, (float) (maxT/2 - 0.1*maxT/2), 0.0f);
			setOutsideFOV(recoIRR, (float) (maxT/2 - 0.1*maxT/2), 0.0f);
			setOutsideFOV(recoWF, (float) (maxT/2 - 0.1*maxT/2), 0.0f);
			setOutsideFOV(recoSD, (float) (maxT/2 - 0.1*maxT/2), 0.0f);
			
			// show the reconstruction results of the different estimation methods
			reco.clone().show("reconstruction of the original sinogram");
			recoLin.clone().show("reconstruction of the Linear Interpolated sinogram");
			recoIRR.clone().show("reconstruction of the Reprojected sinogram");
			recoWF.clone().show("reconstruction of the Wedge-filtered sinogram");
			recoSD.clone().show("reconstrucion of the Spectral-Deconvolution filtered sinogram");
			
			// compute the L2-Norm in relation to the ground truth image (phantom)
			Grid2D recoDiffLin = (Grid2D) NumericPointwiseOperators.subtractedBy(phantom, recoLin);
			Grid2D recoDiffIRR = (Grid2D) NumericPointwiseOperators.subtractedBy(phantom, recoIRR);
			Grid2D recoDiffWF = (Grid2D) NumericPointwiseOperators.subtractedBy(phantom, recoWF);
			Grid2D recoDiffSD = (Grid2D) NumericPointwiseOperators.subtractedBy(phantom, recoSD);
			
			NumericPointwiseOperators.abs(recoDiffLin);
			NumericPointwiseOperators.abs(recoDiffIRR);
			NumericPointwiseOperators.abs(recoDiffWF);
			NumericPointwiseOperators.abs(recoDiffSD);
			
			NumericPointwiseOperators.multiplyBy(recoDiffLin, recoDiffLin);
			NumericPointwiseOperators.multiplyBy(recoDiffIRR, recoDiffIRR);
			NumericPointwiseOperators.multiplyBy(recoDiffWF, recoDiffWF);
			NumericPointwiseOperators.multiplyBy(recoDiffSD, recoDiffSD);
			
			double 	l2errorLin = NumericPointwiseOperators.sum(recoDiffLin),
					l2errorIRR = NumericPointwiseOperators.sum(recoDiffIRR),
					l2errorWF = NumericPointwiseOperators.sum(recoDiffWF),
					l2errorSD = NumericPointwiseOperators.sum(recoDiffSD);
			
			l2errorLin /= recoDiffLin.getNumberOfElements();
			l2errorIRR /= recoDiffIRR.getNumberOfElements();
			l2errorWF /= recoDiffWF.getNumberOfElements();
			l2errorSD /= recoDiffSD.getNumberOfElements();
			
			l2errorLin = Math.sqrt(l2errorLin);
			l2errorIRR = Math.sqrt(l2errorIRR);
			l2errorWF = Math.sqrt(l2errorWF);
			l2errorSD = Math.sqrt(l2errorSD);
			
			l2errorLin /= 1.0;
			l2errorIRR /= 1.0;
			l2errorWF /= 1.0;
			l2errorSD /= 1.0;
			
			// print the L2-Norm results
			System.out.println("L2-Norm of Error Wedge-filtered " + l2errorWF);
			System.out.println("L2-Norm of Error IRR Interpolated " + l2errorIRR);
			System.out.println("L2-Norm of Error Linear Interpolated " + l2errorLin);
			System.out.println("L2-Norm of Error SD Interpolated " + l2errorSD);
			
			System.out.println("DONE");
		}
	}
	
	public static void setOutsideFOV(Grid2D grid, float r, float newVal){
		int 	xcenter = grid.getWidth()/2,
				ycenter = grid.getHeight()/2;
		for(int i = 0; i < grid.getHeight(); ++i){
			for(int j = 0; j < grid.getWidth(); ++j){
				if((Math.pow(i - ycenter, 2) + Math.pow(j - xcenter, 2)) > (Math.pow(r, 2)))
					grid.setAtIndex(j, i, newVal);
			}
		}
	}
}

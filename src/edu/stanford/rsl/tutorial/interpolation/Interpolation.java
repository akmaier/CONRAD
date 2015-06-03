/*
 * Copyright (C) 2014 Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.interpolation;

import java.util.ArrayList;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.BinaryWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.CompensationWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.fan.redundancy.SilverWeights;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.filters.SheppLoganKernel;
import edu.stanford.rsl.tutorial.fourierConsistency.wedgefilter.DoubleWedgeFilterFanES;


/**
 * This class implements different methods that can be used to estimate
 * missing projections of a CT scan. The input parameter g represents the 
 * under-sampled sinogram, that already holds missing projections, w 
 * is the missing projection image, where 0:=missing projection
 * and 1:=measured projection.
 *  
 * @author Marcel Pohlmann
 * 
 */
public abstract class Interpolation {

		/**
		 * Method that in-paints missing projections into the sparse sinogram (parallel-beam geometry), by 
		 * iteratively reprojecting done in the projection space as described in the Paper:
		 * "Projection Space Iteration Reconstruction-Reprojection", 
		 * J. H. Kim, K. Y. KWAK, IEEE Transactions on Medical Imaging, Vol. MI-4, No.3, September 1985
		 * This method assumes a parallel beam sinogram as input parameter!
		 * 
		 * @param g	Sparse sinogram
		 * @param w	Missing projection mask
		 * @param params Fan-Beam parameters
		 * @param maxIter Number of maximum iterations
		 * @return In-painted sinogram
		 */
		public static Grid2D PSIRR(Grid2D g, Grid2D w, double rp, int maxIter, FanParameters params){
			// TODO: CODE STILL BUGGY!
			boolean debug = false;
			
			if(debug){
				new ImageJ();
			}
			
			double 	maxT = params.getMaxT(),
					deltaT = params.getDeltaT();
			
			double 	delta_l = g.getSpacing()[0],
					delta_theta = g.getSpacing()[1];
			
			Grid2D map = new Grid2D(g.getWidth(), g.getHeight());
			Grid2D g_updated = (Grid2D) g.clone();
			
			for(int i = 0; i < maxIter; ++i){
				Grid2D g_filt = (Grid2D) g_updated.clone();
				
				RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
				for (int theta = 0; theta < g_filt.getSize()[1]; ++theta) {
					ramLak.applyToGrid(g_filt.getSubGrid(theta));
				}
				
				for(int m = 0; m < w.getHeight(); ++m){
					if(w.getAtIndex(0, m) == 0.0){
						// found missing projection
						for(int n = 0; n < g.getWidth(); ++n){
							double theta_b = Math.acos(((n*delta_l - (maxT/2))/rp));
							double newVal = 2*rp*Math.sin(theta_b)*g_filt.getAtIndex(n, m)*delta_l*delta_theta;
							double sum = 0.0;
							
							for(int j = 0; j < g.getHeight(); ++j){
								int n1 = (int) Math.round(((rp*Math.cos(m*delta_theta - j*delta_theta + theta_b) + (maxT/2))/delta_l)),
									n2 = (int) Math.round(((rp*Math.cos(m*delta_theta - j*delta_theta - theta_b) + (maxT/2))/delta_l));
								
								if(debug){
									map.setAtIndex(n1, j, 1.0f);
									map.setAtIndex(n2, j, 2.0f);
								}
								
								for(int k = n1; k < n2; ++k){
									if(j !=k){
										sum += (1/(Math.sin(m*delta_theta - j*delta_theta)))*g_filt.getAtIndex(k, j);
									}
								}
								
							}
							newVal += sum*delta_theta*delta_l;
							g_updated.setAtIndex(n, m, (float) newVal);
							if(debug && (n%100 == 0)){
								map.clone().show(); 
							}
							map = new Grid2D(g.getWidth(), g.getHeight());
						}
						//ramLak.applyToGrid(g_updated.getSubGrid(m));
					}
				}
				for(int p = 0; p < g_updated.getHeight(); p++) {
					for(int q = 0; q < g_updated.getWidth(); q++) {
						if(Float.isNaN(g_updated.getAtIndex(q, p))){
							g_updated.setAtIndex(q, p, 0.0f);
						}
					}
				}
			}
			
			
			return g_updated;
		}
		
		/**
		 * Method that in-paints missing projections into the sparse sinogram, by iteratively reprojecting as 
		 * described in the Paper:
		 * "Iterative Reconstruction-Reprojection: An Algorithm for Limited Data Cardiac-Computed Tomography", 
		 * M. Nassi, IEEE Transactions on Biomedical Engineering, Vol. BME-29, No.5, May 1982
		 * @param g	Sparse sinogram
		 * @param w	Missing projection mask
		 * @param params Fan-Beam parameters
		 * @param maxIter Number of maximum iterations
		 * @param gpuDriven Enable OpenCL projector and backprojector
		 * @param imSize Image size of the reconstruction
		 * @param imSpacing Image spacing of the reconstruction
		 * @return In-painted sinogram
		 */
		public static Grid2D IRR(Grid2D g, Grid2D w, FanParameters params, int maxIter, boolean gpuDriven, int[] imSize, float[] imSpacing) {
			boolean debug = false;
			
			// extract FanParameters container
			double 	maxT = params.getMaxT(), 
					deltaT = params.getDeltaT(),
					focalLength = params.getFocalLength(),
					maxBeta = params.getMaxBeta(),
					deltaBeta = params.getDeltaBeta();
			
			// extract image size and spacing for the reconstruction
			int imgSzXMM = imSize[0], // [mm]
				imgSzYMM = imSize[1]; // [mm]
			float 	pxSzXMM = imSpacing[0],
					pxSzYMM = imSpacing[1];
			
			// make instance of backprojector
			FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength,
					deltaT, deltaBeta, imgSzXMM, imgSzYMM);
			
			// make instance of projector
			FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(
					focalLength, maxBeta, deltaBeta, maxT, deltaT);
			
			double[] spacing = {deltaT, deltaBeta};
			
			g.setSpacing(spacing);
			
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
			
			Grid2D reference = (Grid2D) g.clone();
			
			for(int iter = 0; iter < maxIter; ++iter){
				// apply redundancy weights
				Grid2D gFilt = (Grid2D) NumericPointwiseOperators.multipliedBy(g, RedundancyWeights);
				gFilt.setSpacing(spacing);
				
				// apply SheppLogan kernel
				SheppLoganKernel sheppLogan = new SheppLoganKernel((int) (maxT / deltaT), deltaT);
				for (int j=0; j < gFilt.getHeight(); j++){
					sheppLogan.applyToGrid(gFilt.getSubGrid(j));
				}
				
				// do the backprojection
				Grid2D reco = fbp.backprojectPixelDriven(gFilt);
				
				// set negative values caused by the backprojection to zero
				NumericPointwiseOperators.removeNegative(reco);
				
				// set origin and spacing
				reco.setOrigin(-imgSzXMM/2, -imgSzYMM/2);
				reco.setSpacing(pxSzXMM, pxSzYMM);
				
				Grid2D gHat;
				
				// do the forward projection
				if(gpuDriven == true){
					gHat = fanBeamProjector.projectRayDrivenCL(reco);
				}else{
					gHat = fanBeamProjector.projectRayDriven(reco);
				}
				
				// insert current estimations of missing projections
				insertionHelper(g, gHat, w);
				
				if(debug && iter == (maxIter - 1)){
					reference.clone().show("reference sinogram - should look like sparse sino");
					g.clone().show("sparse sinogram inpainted with proj estimates");
					reco.clone().show("latest reconstruction");
					gHat.clone().show("FP of latest reconstruction");
					
					Grid2D g_test = (Grid2D) g.clone();
					
					NumericPointwiseOperators.multiplyBy(g_test, RedundancyWeights);
					
					for (int j=0; j < gFilt.getHeight(); j++){
						sheppLogan.applyToGrid(g_test.getSubGrid(j));
					}
					Grid2D test = fbp.backprojectPixelDrivenCL(g_test);
					NumericPointwiseOperators.removeNegative(test);
					test.clone().show("DEBUGGING");
				}
			}
			
			return g;
		}
		
		/**
		 * Method that estimates missing projections by iteratively applying a wedge-filter in the Fourier Space.
		 * 
		 * @see class DoubleWedgeFilterFanES
		 * @param g Sparse sinogram
		 * @param w Missing projection mask (0 denotes a missing projection)
		 * @param maxIter Number of maximum iterations
		 * @param zeroPad Apply zero-padding before FFT
		 * @param fanParams Array containing maximum object radius rp, source-isocenter distance L,  detector-isocenter distance D
		 * @return sinogram holding missing projections
		 */
		public static Grid2D iterativeWedgeFilter(Grid2D g, Grid2D w, int maxIter, boolean zeroPad, double[] fanParams) {
			// extract fan beam parameters
			double rp = fanParams[0];
			double _L = fanParams[1];
			double _D = fanParams[2];
			
			// initialize Complex Grid2D that will hold the Fourier transform of the sinogram
			Grid2DComplex c = new Grid2DComplex(g, zeroPad);
			
			// compute and set frequency axis
			int[] size = c.getSize();
			double[] spatialSpacing = g.getSpacing();
			double[] fMax = {0.5*(1/spatialSpacing[0]), 0.5*(1/spatialSpacing[1])};
			double[] fSpacing = {2*fMax[0]/size[0], 2*fMax[1]/size[1]};
			double[] fOrigin = {-fMax[0]+fSpacing[0], -fMax[1]+fSpacing[1]};
			
			// compute the double wedge filter
			DoubleWedgeFilterFanES wedgeFilter = new DoubleWedgeFilterFanES(c.getSize(), fSpacing, fOrigin, rp, _L, _D);
			
			// erode wedge filter to avoid discretization problems
			wedgeFilter.erode(7);
			
			for(int iter = 0; iter < maxIter; ++iter) {
				// compute 2D-Fourier transform of the current estimate of the sinogram
				Grid2DComplex cCopy = c.clone();
				cCopy.transformForward();
				
				// FFT-shift the 2D-FT to fit the double wedge filter axis', where the zero frequency is centered
				cCopy.fftshift();
				
				// apply wedge filter on 2D-FT
				for(int i = 0; i < wedgeFilter.getHeight(); ++i) {
					for(int j = 0; j < wedgeFilter.getWidth(); ++j) {
						if(wedgeFilter.getPixelValue(j, i) > 0.0) {
						}else {
							cCopy.putPixelValue(2*j, i, 0.0);
							cCopy.putPixelValue(2*j+1, i, 0.0);
						}
					}
				}
				
				// invert the FFT-shift
				cCopy.ifftshift();
				
				// apply the inverse 2D-FT
				cCopy.transformInverse();
				Grid2D magnitude = cCopy.getMagnSubGrid(0, 0, g.getWidth(), g.getHeight());
				
				// c2 now holds estimated projections; 
				// note that the measured projections are also affected by the double wedge, therefore ..
				Grid2DComplex c2 = new Grid2DComplex(magnitude, zeroPad);
				
				// .. insert ONLY the new projection estimates
				insertionHelper(c, c2, w);
			}
			
			return c.getMagnSubGrid(0, 0, g.getWidth(), g.getHeight());
		}
		
		/**
		 * Method that estimates missing projections by iteratively decreasing the current rp and applying a wedge-filter
		 * in the Fourier Space
		 * 
		 * @see class DoubleWedgeFilterFanES
		 * @param g Sparse sinogram
		 * @param w Missing projection mask (0 denotes a missing projection)
		 * @param maxIter Number of maximum iterations
		 * @param zeroPad Apply zero-padding before FFT
		 * @param fanParams Array containing maximum object radius rp, source-isocenter distance L,  detector-isocenter distance D
		 * @return In-painted sinogram
		 */
		public static Grid2D iterativeWedgeFilter_opti(Grid2D g, Grid2D w, int maxIter, boolean zeroPad, double[] fanParams) {
			// extract Fan Beam Parameters
			double rp = fanParams[0];
			double _L = fanParams[1];
			double _D = fanParams[2];
			
			// initialize output Grid2D as the same size as g
			Grid2D output = new Grid2D(g.getWidth(), g.getHeight());
			
			// define step size for increasing rp
			double delta_rp = rp/2;
			
			// iterate over decreasing rp
			for(double cur_rp = rp; cur_rp > 0.0; cur_rp -= delta_rp){
				Grid2D g_original = (Grid2D) g.clone();
				
				// initialize Complex Grid2D that will hold the Fourier transform of the sinogram
				Grid2DComplex c = new Grid2DComplex(g_original, zeroPad);
				
				// compute and set frequency axis
				int[] size = c.getSize();
				double[] spatialSpacing = g.getSpacing();
				double[] fMax = {0.5*(1/spatialSpacing[0]), 0.5*(1/spatialSpacing[1])};
				double[] fSpacing = {2*fMax[0]/size[0], 2*fMax[1]/size[1]};
				double[] fOrigin = {-fMax[0]+fSpacing[0], -fMax[1]+fSpacing[1]};
				
				// compute the double wedge filter
				DoubleWedgeFilterFanES wedgeFilter = new DoubleWedgeFilterFanES(c.getSize(), fSpacing, fOrigin, cur_rp, _L, _D);
				
				// erode wedge filter to avoid errors caused by discretization
				wedgeFilter.erode(7);

				for(int iter = 0; iter < maxIter; ++iter) {
					// compute 2D-Fourier transform of the current estimate of the sinogram
					Grid2DComplex cCopy = c.clone();
					cCopy.transformForward();
					
					// FFT-shift the 2D-FT to fit the double wedge filter axis', where the zero frequency is centered
					cCopy.fftshift();
					
					// apply wedge filter on 2D-FT
					for(int i = 0; i < wedgeFilter.getHeight(); ++i) {
						for(int j = 0; j < wedgeFilter.getWidth(); ++j) {
							if(wedgeFilter.getPixelValue(j, i) > 0.0) {
							}else {
								cCopy.putPixelValue(2*j, i, 0.0);
								cCopy.putPixelValue(2*j+1, i, 0.0);
							}
						}
					}
					
					// invert the FFT-shift
					cCopy.ifftshift();
					
					// apply the inverse 2D-FT
					cCopy.transformInverse();
					Grid2D magnitude = cCopy.getMagnSubGrid(0, 0, g.getWidth(), g.getHeight());
					
					// c2 now holds estimated projections; 
					// note that the measured projections are also affected by the double wedge, therefore ..
					Grid2DComplex c2 = new Grid2DComplex(magnitude, zeroPad);
					
					// .. insert ONLY the new projection estimates
					insertionHelper(c, c2, w);
				}
				
				Grid2D c_magnitude = c.getMagnSubGrid(0, 0, g.getWidth(), g.getHeight());
				
				// fill estimated projections into corresponding positions in the sinogram
				for(int theta = 0; theta < output.getHeight(); ++theta){
					for(int s = (int) ((output.getWidth()/2) - cur_rp); s <= ((output.getWidth()/2) + cur_rp); s++){
						output.setAtIndex(s, theta, c_magnitude.getAtIndex(s, theta));
					}
				}
				
			}
			
			return output;
		}
		
		/**
		 * Method that fixes image defects by patchwise Spectral Deconvolution as decribed in the paper:
		 * "Defect Interpolation in Digital Radiography - How Object-Oriented Transform Coding Helps", 
		 * T. Aach, V. Metzler, SPIE Vol. 4322: Medical Imaging, February 2001
		 * 
		 * @see method Grid2D SpectralDeconvoltion(..)
		 * @param g_original Image with defect
		 * @param w	Defect mask (0 denotes the defect)
		 * @param patchSize Holds the desired size of the patch
		 * @param maxIter Number of maximum iterations
		 * @param zeroPadSignal Apply zero-padding on image before FFT
		 * @return In-painted image
		 */
		public static Grid2D ApplyPatchwiseSpectralDeconvolution(Grid2D g_original, Grid2D w, int[] patchSize, int maxIter, 
				boolean zeroPadSignal){
			
			Grid2D g = (Grid2D) g_original.clone();
			
			// move patch over image with an overlay of half the patch size (in both directions)
			for(int i = 0; i < g.getHeight() - (patchSize[1]/2); i += (patchSize[1]/2)){
				for(int j = 0; j < g.getWidth() - (patchSize[0]/2); j += (patchSize[0]/2)){
					// patch will hold the current image sector
					Grid2D patch = new Grid2D(patchSize[0], patchSize[1]);
					
					// patch mask will hold the corresponding sector of the defect mask
					Grid2D patch_mask = new Grid2D(patchSize[0], patchSize[1]);
					
					// fill patch and patch_mask with values
					for(int l = 0; l < patchSize[0]; ++l){
						for(int k = 0; k < patchSize[1]; ++k){
							patch.setAtIndex(l, k, g_original.getAtIndex(l + j, k + i));
							patch_mask.setAtIndex(l, k, w.getAtIndex(l + j, k + i));	
						}
					}
					
					// apply Spectral Deconvolution on the current patch
					Grid2D result = SpectralDeconvoltion((Grid2D) patch, (Grid2D) patch_mask, maxIter, zeroPadSignal);
					
					// write the result back into g
					for(int l = 0; l < patchSize[0]; ++l){
						for(int k = 0; k < patchSize[1]; ++k){
							g.setAtIndex(l + j, k + i, result.getAtIndex(l, k));
						}
					}
				}
			}
			
			return g;
		}
		
		/**
		 * Method that fixes image defects by Spectral Deconvolution as decribed in the paper:
		 * "Defect Interpolation in Digital Radiography - How Object-Oriented Transform Coding Helps", 
		 * T. Aach, V. Metzler, SPIE Vol. 4322: Medical Imaging, February 2001
		 * 
		 * @param g_original Image with defect
		 * @param w	Defect mask (0 denotes the defect)
		 * @param maxIter Number of maximum iterations
		 * @param zeroPadSignal Apply zero-padding on image before FFT
		 * @return In-painted image
		 */
		public static Grid2D SpectralDeconvoltion(Grid2D g_original, Grid2D w, int maxIter, boolean zeroPadSignal) {
			Grid2D g = new Grid2D(g_original);
			
			double maxDeltaE_G_Ratio = Double.POSITIVE_INFINITY;
			double maxDeltaE_G_Ratio_Tres = 1.0e-6;
			
			Grid2DComplex G = new Grid2DComplex(g, zeroPadSignal);
			G.transformForward();
			
			Grid2DComplex W = new Grid2DComplex(w, zeroPadSignal);
			W.transformForward();
			
			int[] dim = G.getSize();
			int[] halfDim = {dim[0]/2, dim[1]/2};
			
			// Initialization
			Grid2DComplex FHat = new Grid2DComplex(dim[0], dim[1], false);
			Grid2DComplex FHatNext = new Grid2DComplex(dim[0], dim[1], false);
			
			double lastEredVal = 0;
			
			for(int i = 0; i < maxIter; i++) {
				// Check convergence criterion
				if(maxDeltaE_G_Ratio <= maxDeltaE_G_Ratio_Tres) {
					System.out.println("Break in " + i + "th iteration");
					break;
				}
				// In the i-th iteration select the line pair s1,t1
				// which maximizes the energy reduction [Paragraph after Eq. (16) in the paper]
				double maxDeltaE_G = Double.NEGATIVE_INFINITY;
				ArrayList<Integer[]> sj1 = null;
				// Find the entries with the maximum values and write them in sj1
				for(int j = 0; j < dim[0]; j++) {
					for(int k = 0; k < dim[1]; k++) {
							double val = G.getAtIndex(j, k);
							if(val > maxDeltaE_G) {
								sj1 = new ArrayList<Integer[]>();
								sj1.add(new Integer[]{j,k});
								maxDeltaE_G = val;
							}else if(val == maxDeltaE_G) {
								sj1.add(new Integer[]{j, k});
							}
					}
				}
				int idx = (int) Math.floor(Math.random() * sj1.size());
				int s1 = sj1.get(idx)[0];
				int t1 = sj1.get(idx)[1];
				
				// Calculate the ratio of energy reduction in comparison to the last iteration
				if(i > 0) {
					maxDeltaE_G_Ratio = Math.abs((maxDeltaE_G - lastEredVal) / maxDeltaE_G);
				}
				
				// Save the last energy reduction value for next iteration
				lastEredVal = maxDeltaE_G;
				
				// Compute the corresponding linepair s2, t2:
				// mirror the positions at halfDim
				int s2 = (s1 > 0) ? dim[0] - (s1 % dim[0]) : s1;
				int t2 = (t1 > 0) ? dim[1] - (t1 % dim[1]) : t1;
				
				// This we require in the next step:
			    // [Paragraph after Eq. (17) in the paper]
				int twice_s1 = (2 * s1) % dim[0];
				int twice_t1 = (2 * t1) % dim[1];
				
				boolean specialCase = false;
				
				if(	(s1 == 0 			&& t1 == 0) 		|| 
					(s1 == 0 			&& t1 == halfDim[1])||
					(s1 == halfDim[0] 	&& t1 == 0) 		||
					(s1 == halfDim[0] 	&& t1 == halfDim[1])) {
					//SPECIAL CASES
					
					specialCase = true;
					Complex FHatNextVal = new Complex(FHatNext.getRealAtIndex(s1, t1), FHatNext.getImagAtIndex(s1, t1));
					Complex GVal = new Complex(G.getRealAtIndex(s1, t1), G.getImagAtIndex(s1, t1));
					Complex WVal = new Complex(W.getRealAtIndex(0, 0), W.getImagAtIndex(0, 0));
					Complex res = FHatNextVal.add(GVal.mul(dim[0] * dim[1]).div(WVal));
					
					FHatNext.setRealAtIndex(s1, t1, (float) res.getReal());
					FHatNext.setImagAtIndex(s1, t1, (float) res.getImag());
				}else {
					// GENERAL CASE
					
					Complex FHatNextVal_s1t1 = new Complex(FHatNext.getRealAtIndex(s1, t1), FHatNext.getImagAtIndex(s1, t1));
					Complex FHatNextVal_s2t2 = new Complex(FHatNext.getRealAtIndex(s2, t2), FHatNext.getImagAtIndex(s2, t2));
					Complex GVal = new Complex(G.getRealAtIndex(s1, t1), G.getImagAtIndex(s1, t1));
					Complex WVal00 = new Complex(W.getRealAtIndex(0, 0), W.getImagAtIndex(0, 0));
					Complex WValTwice = new Complex(W.getRealAtIndex(twice_s1, twice_t1), W.getImagAtIndex(twice_s1, twice_t1));
					
					Complex tVal = ((GVal.mul(WVal00)).sub((GVal.getConjugate().mul(WValTwice)))).mul((dim[0] * dim[1]));
					tVal = tVal.div(WVal00.getMagn() * WVal00.getMagn() - WValTwice.getMagn() * WValTwice.getMagn());
					
					Complex res1 = FHatNextVal_s1t1.add(tVal);
					Complex res2 = FHatNextVal_s2t2.add(tVal.getConjugate());
					
					FHatNext.setRealAtIndex(s1, t1, (float) res1.getReal());
					FHatNext.setImagAtIndex(s1, t1, (float) res1.getImag());
					FHatNext.setRealAtIndex(s2, t2, (float) res2.getReal());
					FHatNext.setImagAtIndex(s2, t2, (float) res2.getImag());
				}
				
				// Form the new error spectrum
				updateSpectrum(G, FHatNext, FHat, W, s1, t1, specialCase);
				
				G.setAtIndex(s1, t1, 0);
				if(!specialCase) {
					G.setAtIndex(s2, t2, 0);
				}
				
				FHat = new Grid2DComplex(FHatNext);
			}
			
			// Compute the inverse Fourier Transform of the estimated image
			FHat.transformInverse();
			
			// Insert missing pixels
			for(int j = 0; j < g.getSize()[1]; j++) {
				for(int i = 0; i < g.getSize()[0]; i++) {
					if(w.getAtIndex(i, j) == 0) {
						g.setAtIndex(i, j, FHat.getRealAtIndex(i, j));
					}
				}
			}
			
			return g;
		}
		
		private static void insertionHelper(Grid2D c, Grid2D cHat , Grid2D w) {
			// replace missing lines in c with samples of cHat
			for(int i = 0; i < w.getHeight(); ++i) {
				for(int j = 0; j < w.getWidth(); ++j){
					if(w.getAtIndex(j, i) == 0.0) {
						c.setAtIndex(j, i, cHat.getAtIndex(j, i));
					}
				}
			}
		}
		
		private static void updateSpectrum(Grid2DComplex G, Grid2DComplex FHatNext, Grid2DComplex FHat, Grid2DComplex W, int s1, int t1, boolean specialCase) {
			int[] sz = FHatNext.getSize();
			
			// Accumulation: Update pair (s1,t1),(s2,t2)
			Complex Fst = new Complex(FHatNext.getRealAtIndex(s1, t1) - FHat.getRealAtIndex(s1, t1), FHatNext.getImagAtIndex(s1, t1) - FHat.getImagAtIndex(s1, t1));
			Complex Fstc = Fst.getConjugate();
			
			int divNr = sz[0] * sz[1];
			
			// Compute the new error spectrum
			for(int j = 0; j < sz[1]; j++) {
				for(int i = 0; i < sz[0]; i++) {
					if(specialCase) {
						int xneg = (i - s1) % sz[0];
						int yneg = (j - t1) % sz[1];
						
						if(xneg < 0) {
							xneg = sz[0] + xneg;
						}
						if(yneg < 0) {
							yneg = sz[1] + yneg;
						}
						
						Complex GVal = new Complex(G.getRealAtIndex(i, j), G.getImagAtIndex(i, j));
						G.setRealAtIndex(i, j, (float) GVal.getReal());
						G.setImagAtIndex(i, j, (float) GVal.getImag());
					}else {
						int xpos = (i + s1) % sz[0];
						int ypos = (j + t1) % sz[1];
						int xneg = (i - s1) % sz[0];
						int yneg = (j - t1) % sz[1];
						
						if(xneg < 0) {
							xneg = sz[0] + xneg;
						}
						if(yneg < 0) {
							yneg = sz[1] + yneg;
						}
						
						Complex WPos = new Complex(W.getRealAtIndex(xpos, ypos), W.getImagAtIndex(xpos, ypos));
						Complex WNeg = new Complex(W.getRealAtIndex(xneg, yneg), W.getImagAtIndex(xneg, yneg));
						Complex GVal = new Complex(G.getRealAtIndex(i, j), G.getImagAtIndex(i, j));
						GVal = GVal.sub(((Fst.mul(WNeg)).add(Fstc.mul(WPos))).div(divNr));
						
						G.setRealAtIndex(i, j, (float) GVal.getReal());
						G.setImagAtIndex(i, j, (float) GVal.getImag());
					}
				}
			}
		}

}

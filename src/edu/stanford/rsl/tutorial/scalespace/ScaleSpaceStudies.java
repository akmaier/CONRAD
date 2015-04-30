package edu.stanford.rsl.tutorial.scalespace;

import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;
import ij.ImageJ;

/**
 * Class to convolve projection images of a phantom with either a Gaussian or a Laplacian of Gaussian.
 * Different values for sigma are used to show the impact of their size.
 * 
 * @author Markus Wolf
 */

public class ScaleSpaceStudies extends Grid2D {
	
	
	public static void main(String args[]) {
		
		new ImageJ();
		
		// phantom respectively image parameters
		int phantomType = 0; // 0 = own created phantom
							 // 1 = SheppLogan
							 // 2 = UniformCircleGrid2D
		
		int phantomWidth = 256,
			phantomHeight = 256;
		
		int imgSizeX = phantomWidth,
			imgSizeY = phantomHeight;
		
		// parameters for Gaussian/Laplacian of Gaussian
		int kernelsize = 10;
		
		int method = 2; // 0 = Gaussian
						// 2 = Laplacian of Gaussian
			
		// HINT: Since sigma is used differently in Gaussian and Laplacian of Gaussian, different values are necessary.
		// 		 Too high sigma values for Gaussian (method 0) results in NaN in the resulted image!
		double a, b, c, d;
		
		switch (method) {
		case 0:			
			a = 0.1;
			b = 0.5;
			c = 1.0;
			d = 2.0;
			break;
		case 2:
			a = 0.1;
			b = 1.0;
			c = 3.0;
			d = 5.0;
			break;
		default:
			a = 0.1;
			b = 1.0;
			c = 3.0;
			d = 5.0;
		}
		double[] sigmaValue = {a, b, c, d};
		
		// fan beam bp parameters
		double gammaM = 11.768288932020647*Math.PI/180,
			   maxT = (int)Math.round(Math.sqrt((phantomWidth*phantomWidth) + (phantomHeight*phantomHeight))),
			   deltaT = 1.0, 
			   focalLength = (maxT/2.0-0.5)*deltaT/Math.tan(gammaM),
		   	   maxBeta = 360*Math.PI/180,
		       deltaBeta = maxBeta / 180;
		
		// create phantom		
		Grid2D phantom;
		
		switch(phantomType) {
		case 0:
			phantom = new ScaleSpaceStudies(phantomWidth, phantomHeight);
			break;
		case 1:
			phantom = new SheppLogan(phantomWidth,false);
			break;
		case 2:
			phantom = new UniformCircleGrid2D(phantomWidth, phantomHeight);
			break;
		default:
			phantom = new SheppLogan(phantomWidth,false);
		}

		phantom.show("phantom");

		// create projections (sinogram) out of the phantom
		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
				
		Grid2D projectionP = new Grid2D(phantom);
		Grid2D fanBeamSinoRay = fanBeamProjector.projectRayDrivenCL(projectionP);	
		fanBeamSinoRay.show("sinogram");

		// calculate Gaussian or Laplacian of Gaussian on projection images for different values of sigma
		for (int i = 0; i < sigmaValue.length; i++) {
			
			Grid2D workingGrid;
			
			switch (method){
			case 0:			
				workingGrid = gaussianSinogram(fanBeamSinoRay, sigmaValue[i]);
				workingGrid.show("gaussian (sigma=" + sigmaValue[i] + ") sinogram");
				break;
			case 2:
				workingGrid = laplacianOfGaussianSinogram(fanBeamSinoRay, kernelsize, sigmaValue[i]);
				workingGrid.show("laplacian of gaussian (sigma=" + sigmaValue[i] + ") sinogram");
				break;
			default:
				workingGrid = laplacianOfGaussianSinogram(fanBeamSinoRay, kernelsize, sigmaValue[i]);
				workingGrid.show("laplacian of gaussian (sigma=" + sigmaValue[i] + ") sinogram");
			}
			
			// correct the sinograms	
			RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
			CosineFilter cKern = new CosineFilter(focalLength, maxT, deltaT);
			
			// apply filtering
			for (int theta = 0; theta < workingGrid.getSize()[1]; ++theta) {
				cKern.applyToGrid(workingGrid.getSubGrid(theta));
			}
		
			for (int theta = 0; theta < workingGrid.getSize()[1]; ++theta) {
				ramLak.applyToGrid(workingGrid.getSubGrid(theta));
			}
						
			// do the backprojection
			FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength, deltaT, deltaBeta, imgSizeX, imgSizeY);
			
			Grid2D recoWorkingGrid;
			
			switch (method) {
			case 0:
				recoWorkingGrid = fbp.backprojectPixelDrivenCL(workingGrid);
				recoWorkingGrid.show("gaussian (sigma=" + sigmaValue[i] + ") reconstruction");
				break;
			case 2:
				recoWorkingGrid = fbp.backprojectPixelDrivenCL(workingGrid);
				recoWorkingGrid.show("laplacian of gaussian (sigma=" + sigmaValue[i] + ") reconstruction");
				break;
			default:
				recoWorkingGrid = fbp.backprojectPixelDrivenCL(workingGrid);
				recoWorkingGrid.show("laplacian of gaussian (sigma=" + sigmaValue[i] + ") reconstruction");
			}
						
			// difference between phantom and reconstruction
			// HINT: Only makes sense for Gaussian.
			if (method == 0) {
				NumericGrid recoDiff = NumericPointwiseOperators.subtractedBy(phantom, recoWorkingGrid);
				recoDiff.show("gaussian (sigma=" + sigmaValue[i] + ") RecoDiff");
			}		
		}
	}
	
	/**
	 * create own phantom
	 */
	public ScaleSpaceStudies(int width, int height){
		super(width, height);
		
		// center of grid
		final double ctr_w = width/2;
		final double ctr_h = height/2;	
		
		// put some concrete shapes in the phantom
		for (int i = 0; i < width; i++){
			for (int j = 0; j < height; j++){
				// big circle
				if(((i-ctr_w-10)*(i-ctr_w-10)+(j-ctr_h-10)*(j-ctr_h-10)) < Math.pow(19, 2)){
				putPixelValue(i,j,1.0);
				}
				// small circle
				if(((i-ctr_w+10)*(i-ctr_w+10)+(j-ctr_h+10)*(j-ctr_h+10)) < Math.pow(1.5, 2)){
					putPixelValue(i,j,1.5);
				}
				// left ellipse
				if((((i-ctr_w+35)*(i-ctr_w+35))/26+((j-ctr_h+20)*(j-ctr_h+20)))/32 < 1){
					putPixelValue(i,j,0.9);
				}
			}
		}
		
		// rectangle
		for (int i = 10; i < ctr_w; i++){
			for (int j = (int)ctr_h; j < height-15; j++){
				if (getPixelValue(i,j) == 0){
					putPixelValue(i,j,1.5);
				} else {
					putPixelValue(i,j, getPixelValue(i,j)+1.5);
				}		
			}
		}
	}

	/**
	 * convolution of sinogram and Gaussian -> multiplication in Fourier Space
	 */
	public static Grid2D gaussianSinogram(Grid2D sinogram, double sigma) {
		Grid2D gausSinogram = new Grid2D(sinogram);
		
		for(int i = 0; i < sinogram.getHeight(); i++){
			Grid1DComplex sinoRows = new Grid1DComplex(sinogram.getSubGrid(i));
			
			// Fourier Transformation
			sinoRows.transformForward();
			
			for(int j = 0; j < sinoRows.getSize()[0]; j++){
				
				double omega = (j - sinoRows.getSize()[0]/2)*(2*Math.PI/sinoRows.getSize()[0]);
				
				float gausValueReal = sinoRows.getRealAtIndex(j)*(float)Math.exp(0.5*Math.pow(sigma, 2)*Math.pow(omega, 2));
				float gausValueImag = sinoRows.getImagAtIndex(j)*(float)Math.exp(0.5*Math.pow(sigma, 2)*Math.pow(omega, 2));
				
				sinoRows.setRealAtIndex(j, gausValueReal);
				sinoRows.setImagAtIndex(j, gausValueImag);
			}
			
			// inverse Fourier Tranformation
			sinoRows.transformInverse();
			
			// copy again into sinogram
			for(int k = 0; k < sinogram.getWidth(); k++){
				float val = (float)Math.sqrt((float)Math.pow(sinoRows.getRealAtIndex(k), 2) + (float)Math.pow(sinoRows.getImagAtIndex(k), 2));
				gausSinogram.setAtIndex(k, i, val);
			}
		}
		
		return gausSinogram;
	}

	/**
	 * convolution of sinogram and Laplacian of Gaussian
	 */
	public static Grid2D laplacianOfGaussianSinogram(Grid2D sinogram, int kernelsize, double sigma) {
		
		Grid2D sinogramLoG = new Grid2D(sinogram);
		
		for(int i = 0; i < sinogram.getHeight(); i++) {
			Grid1DComplex sinoRow = new Grid1DComplex(sinogram.getSubGrid(i));

			Grid1DComplex result = convolutionLoG(sinoRow, kernelsize, sigma);
		
			// copy again into sinogram
			for(int k = 0; k < sinogram.getWidth(); k++){
				sinogramLoG.setAtIndex(k, i, result.getAtIndex(k));
			}
		}	
		
		return sinogramLoG;	
	}

	/**
	 * compute convolution with Laplacian of Gaussian
	 */
	public static Grid1DComplex convolutionLoG(Grid1DComplex signal, int kernelsize, double sigma) {

		int N = signal.getSize()[0];
		Grid1DComplex result = new Grid1DComplex(N);
		
		for (int n = 0; n < N; n++) { // n is number of signal element
			int kmin, kmax;
			float valueConvolved = 0.0f;
			
			result.setAtIndex(n, 0);

			// borders for kernel window
			kmin = (n >= kernelsize - 1) ? n - (kernelsize - 1) : 0;
			kmax = (n < N - 1) ? n : N - 1;

			for (int k = kmin; k <= kmax; k++) {
				valueConvolved = (float) (valueConvolved + signal.getAtIndex(k) * laplacianOfGaussian(n-k, sigma));
			}
			result.setAtIndex(n, valueConvolved);
		}
		
		return result;
	}
	
	/**
	 * Laplacian of Gaussian -> second derivative of Gaussian distribution
	 */
	public static double laplacianOfGaussian(int index, double sigma) {
		
		double value = (-1/(Math.pow(sigma, 3) * Math.sqrt(2*Math.PI))) * (1 - (Math.pow(index, 2) / Math.pow(sigma, 2))) * Math.exp(-(Math.pow(index, 2) / (2*Math.pow(sigma, 2))));
	
		return value;
	}

}

/*
 * Copyright (C) 2010-2015 Markus Wolf
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

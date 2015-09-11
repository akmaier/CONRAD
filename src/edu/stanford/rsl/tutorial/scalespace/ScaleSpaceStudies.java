package edu.stanford.rsl.tutorial.scalespace;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
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
		int phantomType = 4; // 0 = custom phantom
							 // 1 = SheppLogan
							 // 2 = UniformCircleGrid2D
							 // 3 = Ring
							 // 4 = Dartboard
		
		int phantomWidth  = 256,
			phantomHeight = 256;
		
		int imgSizeX = phantomWidth,
			imgSizeY = phantomHeight;
		
		// parameters for Gaussian/Laplacian of Gaussian
		int kernelsize = 30;
		
		int method = 2; // 0 = Gaussian
						// 2 = Laplacian of Gaussian
		
		double[] sigmaValue = {0.1, 1.0, 3.0, 5.0};
		
		// fan beam bp parameters
		double gammaM      = 11.768288932020647*Math.PI/180,
			   maxT        = (int)Math.round(Math.sqrt((phantomWidth*phantomWidth) + (phantomHeight*phantomHeight))),
			   deltaT      = 1.0, 
			   focalLength = (maxT/2.0-0.5)*deltaT/Math.tan(gammaM),
		   	   maxBeta     = 360*Math.PI/180,
		       deltaBeta   = maxBeta / 180;
		
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
		case 3:
			Grid2D outerCircle = new UniformCircleGrid2D(phantomWidth, phantomHeight, 0.4);
			Grid2D innerCircle = new UniformCircleGrid2D(phantomWidth, phantomHeight, 0.37);
			phantom = new Grid2D(outerCircle);
			NumericPointwiseOperators.subtractBy(phantom, innerCircle);
			break;
		case 4:
			phantom = phantomDartboard(phantomWidth, phantomHeight, 0.4);
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
	 * create own phantom consisting of
	 * <ul><li>a big circle with the intensity 1.0,
	 * <li>a small circle with the intensity 1.5,
	 * <li>an ellipse with the intensity 0.9,
	 * <li>a rectangle with the intensity 1.5
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
	 * create phantom that looks like a dartboard with 9 rings
	 */
	public static Grid2D phantomDartboard(int width, int height, double r){
			
		Grid2D dartboard = new Grid2D(width, height);
		
		float[] val = {0.05f, 0.06f, 0.08f, 0.09f, 0.1f, 0.13f, 0.14f, 0.15f, 0.2f};
		double[] radius = {r * width, (r-0.03)*width, (r-0.06)*width, (r-0.1)*width, (r-0.2)*width, (r-0.25)*width, (r-0.30)*width, (r-0.35)*width, (r-0.38)*width};
		int xcenter = width / 2;
		int ycenter = height / 2;

		for (int k = 0; k < val.length; k++) {			
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius[k] * radius[k])) {
						dartboard.addAtIndex(i, j, val[k]);
					}
				}
			}
		}
		return dartboard; 
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
			
			for(int j = 0; j < sinoRows.getSize()[0]/2; j++){
				
				double omega = j*(2*Math.PI/sinoRows.getSize()[0]);
				
				float gausValueReal = sinoRows.getRealAtIndex(j)*(float)Math.exp(-0.5*Math.pow(sigma, 2)*Math.pow(omega, 2));
				float gausValueImag = sinoRows.getImagAtIndex(j)*(float)Math.exp(-0.5*Math.pow(sigma, 2)*Math.pow(omega, 2));
				
				sinoRows.setRealAtIndex(j, gausValueReal);
				sinoRows.setImagAtIndex(j, gausValueImag);
			}
			
			for(int j = sinoRows.getSize()[0]/2; j < sinoRows.getSize()[0]; j++){
				
				double omega = (j - sinoRows.getSize()[0])*(2*Math.PI/sinoRows.getSize()[0]);
				
				float gausValueReal = sinoRows.getRealAtIndex(j)*(float)Math.exp(-0.5*Math.pow(sigma, 2)*Math.pow(omega, 2));
				float gausValueImag = sinoRows.getImagAtIndex(j)*(float)Math.exp(-0.5*Math.pow(sigma, 2)*Math.pow(omega, 2));
				
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
			Grid1D sinoRow = new Grid1D(sinogram.getSubGrid(i));

			Grid1D result = convolutionLoG(sinoRow, kernelsize, sigma);
		
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
	public static Grid1D convolutionLoG(Grid1D signal, int kernelsize, double sigma) {

		int N = signal.getSize()[0];
		Grid1D result = new Grid1D(N);

		for (int n = 0; n < N; n++) { // n is number of signal element

			float valueConvolved = 0.0f;			
			result.setAtIndex(n, 0);

			// borders for kernel window
			int kmin, kmax;
			int kernelrad = kernelsize/2;
			kmin = (n > kernelrad) ? n - kernelrad : 0;
			kmax = (n < N - kernelrad) ? n + kernelrad : N - 1;
				
			// local sampling
			float deltaU;			// additional factor (between 0.0 and 0.9) for sampling rate
			float v;				// sampling rate
			float interpolSigVal;	// interpolated signal value
			
			for (int k = kmin; k < kmax; k++) {
				for (int u = 0; u < 10; u++) {
					
					deltaU = 0.1f * u;
					v = k + deltaU;
					
					interpolSigVal = (1-deltaU)*signal.getAtIndex(k) + deltaU*signal.getAtIndex(k+1);	// interpolation due to local sampling
					valueConvolved += interpolSigVal * laplacianOfGaussian(n-v, sigma);					// actual convolution
				}
			}
			
			result.setAtIndex(n, valueConvolved);
		}
		
		return result;
	}
	
	/**
	 * Laplacian of Gaussian -> second derivative of Gaussian distribution
	 */
	public static double laplacianOfGaussian(double index, double sigma) {
		
		double value = (-1/(Math.pow(sigma, 3) * Math.sqrt(2*Math.PI))) * (1 - (Math.pow(index, 2) / Math.pow(sigma, 2))) * Math.exp(-(Math.pow(index, 2) / (2*Math.pow(sigma, 2))));
	
		return value;
	}

}

/*
 * Copyright (C) 2010-2015 Markus Wolf
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

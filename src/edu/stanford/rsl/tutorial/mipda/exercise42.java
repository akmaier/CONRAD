package edu.stanford.rsl.tutorial.mipda;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.MeanFilteringTool;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

/**
 * Exercise 4-2 of Diagnostic Medical Image Processing (DMIP)
 * @author Frank Schebesch
 *
 */

public class exercise42 {
	
	Grid2D originalImage;
	Grid2D biasedImage;
	
	public static void main(String[] args) {

		/* instantiate exercise object */
		exercise42 ex4Object = new exercise42("head"); // "box" or "head"
		
		/* start ImageJ */
		// always start an instance of ImageJ for image analysis
		new ImageJ();

		Grid2D image = ex4Object.originalImage;
		Grid2D biased = ex4Object.biasedImage;
		
		image.show("Original image");
		
		Grid2D corrected = extClone(biased); // careful: reallyClone is implemented only for this exercise!

		biased.show("Biased image with noise");
		
		/** 
		 * preprocessing
		 */
		ImagePlus imp = new ImagePlus("",ImageUtil.wrapGrid2D(biased));
		IJ.run(imp, "Gaussian Blur...", "sigma=1.5"); // imp works on image directly

		
		/**
		 * implement homomorphic unsharp masking 
		 */

		int kernelSize = 31;
		 
		/* 
		 * homomorphic unsharp masking:
		 * - compute global and local means
		 * - weight the image by mu/mu_ij
		 *
		 * TODO
		 */
		Grid2D mu_ij = (Grid2D) biased.clone();

		float mu = ex4Object.meanPositives(biased);
		
		float eps = 0.01f;
		for (int i = 0; i < corrected.getHeight(); i++){
			for (int j = 0; j < corrected.getWidth(); j++){
				
				float tempVal;
				
				// this is a workaround to avoid the quotient to get too large or too small
				if (mu_ij.getAtIndex(j, i) <= eps*mu){
					tempVal = 1.0f/eps;
				}
				else if (mu_ij.getAtIndex(j, i) >= mu/eps){
					tempVal = eps;
				}
				else{
					tempVal = 1.0f; // TODO
				}
				
				// TODO: apply the bias field correction for the multiplicative model
			}
		}
		
		corrected.show("Corrected image using homomorphic unsharp masking");		
		
		
		/**
		 * frequency based correction 
		 */
		Grid2DComplex fftimage = new Grid2DComplex(biased); // check the Grid size and look up zero-padding
		// transform into Fourier domain 
		// TODO
		
		double beta = 1.0d;
		double sigma = 0.01d;
		
		for (int k = 0; k < fftimage.getHeight(); k++){
			for (int l = 0; l < fftimage.getWidth(); l++){	
				
				double fk = (k-fftimage.getHeight()/2.0d)/fftimage.getWidth()/biased.getSpacing()[0];
				double fl = (l-fftimage.getWidth()/2.0d)/fftimage.getHeight()/biased.getSpacing()[1];
				
				/* Apply the reverse Gaussian filter from the lecture
				 * TODO
				 */
			}
		}

		// transform back into spatial domain
		// TODO
		
		Grid2D corrected2 = extClone(biased);
		
		for (int i = 0; i < corrected2.getHeight(); i++){
			for (int j = 0; j < corrected2.getWidth(); j++){
				corrected2.setAtIndex(j, i, fftimage.getRealAtIndex(j, i));
			}
		}
		
		corrected2.show("Corrected image using frequency filtering");	
		
		/** 
		 * for comparison: hard frequency cutoff 
		 */
		Grid2D corrected3 = extClone(biased);
	
		// TODO: apply a hard frequency cut-off, look for the appropriate class method
		corrected3.show("Corrected image frequency cutoff");
		
		IJ.run(imp, "Window/Level...", ""); // open tool to window the images properly
	}
	
	public Grid2D lowPass(Grid2D grid, double cutoff){
		return passFilter(grid,true,cutoff);
	}
	
	public Grid2D highPass(Grid2D grid, double cutoff){
		return passFilter(grid,false,cutoff);
	}
	
	public Grid2D readImageFile(String filename){
		
//		return ImageUtil.wrapImagePlus(IJ.openImage(getClass().getResource(filename).getPath())).getSubGrid(0);
		return ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
	}
	
	private void simulateBiasedImage(Grid2D image, Grid2D biasField){
		NumericPointwiseOperators.multiplyBy(image, biasField);
	}
	
	public void addNoise(Grid2D image, double noiseLevel){
		
		FloatProcessor fp;
		
		fp = ImageUtil.wrapGrid2D(image);
		fp.noise(noiseLevel*(fp.getMax()-fp.getMin())); // direct change of input image
	}
	
	private float meanPositives(Grid2D grid){
		
		int numNonZeros = 0;
		float val = 0;
		
		for (int i = 0; i < grid.getHeight(); i++){
			for (int j = 0; j < grid.getWidth(); j++){
				if (grid.getAtIndex(j, i) > 0){
					val += grid.getAtIndex(j, i);
					numNonZeros++;
				}
			}
		}
		
		val/=numNonZeros;
		
		return val;
	}
	
	private void generateExerciseGainField(Grid2D gainField){
		
		float upPerc = 0.2f;
		float lowPerc = 5.0f;
		
		for (int i = 0; i < gainField.getHeight(); i++){
			for (int j = 0; j < gainField.getWidth(); j++){
				
				double[] coords = gainField.indexToPhysical(j, i);
				
				double fx = 0.5d*coords[0]/gainField.getWidth()*Math.PI;
				double fy = 0.5d*coords[1]/gainField.getHeight()*Math.PI;

				float val = (float) (0.5*fx*fy + 0.3*fx*fx*fx + 0.7*fy*fy + 0.1*fx + 0.4*fy);
				
				gainField.setAtIndex(j, i, val);
			}
		}
		
		scaleGrid(gainField, upPerc, lowPerc);
	}

	private void placeBoxOnGrid(Grid2D grid,double magnitude,double[] mu,double[] bds){
		
		for (int i = 0; i < grid.getHeight(); i++){
			for (int j = 0; j < grid.getWidth(); j++){
				
				double[] coords = grid.indexToPhysical(j,i); // get actual position (not indices)

				if (coords[0]>-bds[0]+mu[0] && coords[0]<bds[0]+mu[0] 
						&& coords[1]>-bds[1]+mu[1] && coords[1]<bds[1]+mu[1])
					grid.setAtIndex(j, i, (float)magnitude);
			}
		}
	}
	
	protected static float get2DGaussianValueAt(double[] coords, double magnitude, double[] mu, double[] sigma){
		
		double factor = magnitude*1.0d/(2.0d*Math.PI)/sigma[0]/sigma[1];
		
		double x2 = Math.pow((coords[0]-mu[0])/sigma[0],2.0d);
		double y2 = Math.pow((coords[1]-mu[1])/sigma[1],2.0d);
		
		return (float) (factor*Math.exp(-(x2+y2)/2.0f));
	}
	
	protected static Grid2D extClone(Grid2D grid){
		
		Grid2D clonedGrid = (Grid2D) grid.clone();
		clonedGrid.setOrigin(grid.getOrigin());
		clonedGrid.setSpacing(grid.getSpacing());
		
		return clonedGrid;
	}
	
	protected static Grid2DComplex extClone(Grid2DComplex grid){
		
		Grid2DComplex clonedGrid = (Grid2DComplex) grid.clone();
		clonedGrid.setOrigin(grid.getOrigin());
		clonedGrid.setSpacing(grid.getSpacing());
		
		return clonedGrid;
	}	
	
	public void generateGaussian2D(Grid2D grid, double[] mu, double[] sigma){
		
		for (int i = 0; i < grid.getHeight(); i++){
			for (int j = 0; j < grid.getWidth(); j++){
				
				double[] coords = grid.indexToPhysical(j,i); // get actual position (not indices)
				
				float val = get2DGaussianValueAt(coords,1.0d,mu,sigma);
				grid.setAtIndex(j, i, val);
			}
		}
	}
	
	private Grid2D passFilter(Grid2D grid, boolean type, double cutoff){
		
		// type=true => low pass
		// type=false => high pass
		
		Grid2DComplex fftGrid = new Grid2DComplex(grid);
		Grid2D filteredGrid = extClone(grid);
		
		fftGrid.transformForward(); // does not know of spacing or origin
		fftGrid.setSpacing(1.0f/fftGrid.getWidth()/grid.getSpacing()[0],1.0f/fftGrid.getHeight()/grid.getSpacing()[1]);
		fftGrid.fftshift();
		
//		fftGrid.getMagnSubGrid(0,0,fftGrid.getWidth(),fftGrid.getHeight()).show("Magnitude of FFT image");
//		fftGrid.getPhaseSubGrid(0,0,fftGrid.getWidth(),fftGrid.getHeight()).show("Phase of FFT image");
		
		for (int i = 0; i < fftGrid.getHeight(); i++){
			for (int j = 0; j < fftGrid.getWidth(); j++){

				double[] frequencies = 
						new double[]{(i-fftGrid.getHeight()/2.0d)*fftGrid.getSpacing()[0],
								(j-fftGrid.getWidth()/2.0d)*fftGrid.getSpacing()[1]};
				if (type){
					if (Math.pow(frequencies[0],2.0d) + Math.pow(frequencies[1],2.0d) > Math.pow(cutoff,2.0d)){
						fftGrid.setRealAtIndex(j, i, 0.0f);
						fftGrid.setImagAtIndex(j, i, 0.0f);	
					}
				}
				else{
					if (Math.pow(frequencies[0],2.0d) + Math.pow(frequencies[1],2.0d) < Math.pow(cutoff,2.0d)){
						fftGrid.setRealAtIndex(j, i, 0.0f);
						fftGrid.setImagAtIndex(j, i, 0.0f);	
					}
				}
				fftGrid.setImagAtIndex(j, i, 0.0f);
			}
		}
		
//		fftGrid.getMagnSubGrid(0,0,fftGrid.getWidth(),fftGrid.getHeight()).show("Magnitude of filtered FFT image");
		
		fftGrid.ifftshift();
		fftGrid.transformInverse();
		
		NumericPointwiseOperators.copy(filteredGrid,fftGrid.getRealSubGrid(0,0,grid.getWidth(),grid.getHeight()));
		
		return filteredGrid;
	}
	
	private void normalize(Grid2D grid){
		
		// Normalize intensity values to [0,1]
		scaleGrid(grid, 0.0f, 1.0f);
	}
	
	private void scaleGrid(Grid2D grid, float newMin, float newMax){
		
		float max = NumericPointwiseOperators.max(grid);
		float min = NumericPointwiseOperators.min(grid);
		
		for(int i = 0; i < grid.getHeight(); i++){
			for(int j = 0; j < grid.getWidth(); j++){
				
				float scaledValue = (grid.getAtIndex(j, i) - min) / (max-min) * (newMax-newMin) + newMin;
				grid.setAtIndex(j, i, scaledValue);
			}
		}
	}
	
	public exercise42(String example){
		
		/* image parameters */
		// define virtual spacing
		float imageXspacing = 1.0f;
		float imageYspacing = 1.0f;
		
		// noise (Gaussian)
		double noiseLevel = 0.01d; // percentage	
		
		/* image container */
		Grid2D gainField;

		/* specify image container properties */
		switch (example) {
		
		case "head":
			
			String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/dmip/";
			
			originalImage = readImageFile(imageDataLoc + "mr_head_dorsal.jpg");
			
			originalImage.setSpacing(imageXspacing,imageYspacing);
			// move origin to center
			originalImage.setOrigin(-originalImage.getWidth()/2.0d*originalImage.getSpacing()[0],
					-originalImage.getHeight()/2.0d*originalImage.getSpacing()[1]);
			
			break;
			
		default:
			originalImage = new Grid2D(256,386);
			
			originalImage.setSpacing(imageXspacing,imageYspacing);
			// move origin to center
			originalImage.setOrigin(-originalImage.getWidth()/2.0d*originalImage.getSpacing()[0],
					-originalImage.getHeight()/2.0d*originalImage.getSpacing()[1]);
			
			placeBoxOnGrid(originalImage,1.0d,new double[]{20.0d,30.0d},new double[]{30.0d,50.0d});
			
			break;
		}

		// Normalize intensity values to [0,1]
		normalize(originalImage);
		
		// copy settings
		gainField = extClone(originalImage);
		
		/* generate biased noisy image */
		generateExerciseGainField(gainField);
		
		biasedImage = extClone(originalImage);
		simulateBiasedImage(biasedImage, gainField);
		
		addNoise(biasedImage,noiseLevel);
	}
}

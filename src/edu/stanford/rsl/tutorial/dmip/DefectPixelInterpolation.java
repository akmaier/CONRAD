package edu.stanford.rsl.tutorial.dmip;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.filtering.MedianFilteringTool;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImageJ;

/**
 * 
 * Exercise 4 of Diagnostic Medical Image Processing (DMIP)
 * @author Marco Boegel
 *
 */
public class DefectPixelInterpolation {
	
	public DefectPixelInterpolation()
	{
		
	}
	
	/**
	 * Spectral deconvolution for defect pixel interpolation
	 * "Defect Interpolation in Digital Radiography - How Object-Oriented Transform Coding Helps", 
	 * T. Aach, V. Metzler, SPIE Vol. 4322: Medical Imaging, February 2001
	 * @param image corrupted image with defect pixels
	 * @param mask binary image mask of the defect pixels (0: defect, 1: fine)
	 * @param maxIter maximum number of iterations
	 * @param zeroPadding enables zero padding for the input images. Images are enlarged to size of next power of 2 and filled with zeros
	 * @return image, where defect pixels are interpolated
	 */
	public Grid2D interpolateSpectral(Grid2D image, Grid2D mask, int maxIter, boolean zeroPadding)
	{
		//padding
		//TODO
		//TODO
		
		//fourier transform
		//TODO
		//TODO
		
		int[] dim = G.getSize();
		
		int[] halfDim = {dim[0]/2, dim[1]/2};
		
		double maxDeltaE_G_Ratio = Double.POSITIVE_INFINITY;
		double maxDeltaE_G_Thresh = 1.0e-6;
		
		Grid2DComplex FHat = new Grid2DComplex(dim[0], dim[1], false);
		Grid2DComplex FHatNext = new Grid2DComplex(dim[0], dim[1], false);
		
		//Setup visualization
		double lastEredVal = 0;
		//Visualize every 100th iteration
		Grid3D visualize = new Grid3D(image.getWidth(), image.getHeight(), maxIter/100);
		
		for(int i = 0; i < maxIter; i++)
		{
			//Check for convergence
			if(maxDeltaE_G_Ratio <= maxDeltaE_G_Thresh)
			{
				System.out.println("maxDeltaE_G_Ratio = " + maxDeltaE_G_Ratio);
				break;
			}
			
			//In the i-th iteration select line pair s1,t1
			//which maximizes the energy reduction [Paragraph after Eq. (16) in the paper]
			double maxDeltaE_G = Double.NEGATIVE_INFINITY;			
			//create arraylist to store lines (in case multiple maxima are found)
			//TODO
			for(int x = 0; x < dim[0]; x++)
			{
				for(int y = 0; y < dim[1]; y++)
				{
					float val = G.getAtIndex(x, y) ;
					if( val > maxDeltaE_G)
					{
						//TODO
						//TODO
						//TODO
					}else if(val == maxDeltaE_G) {
						//TODO
					}
				}
			}
			//if there were more indices than one with the same max_value, pick a random one of these
			int idx = (int) Math.floor(Math.random() * sj1.size());
			int s1 = sj1.get(idx)[0];
			int t1 = sj1.get(idx)[1];

			//Calculate the ratio of energy reduction in comparison to the last iteration
			if(i > 0)
			{
				maxDeltaE_G_Ratio = Math.abs(maxDeltaE_G - lastEredVal/maxDeltaE_G);
			}
			
			lastEredVal = maxDeltaE_G;
			
			//Compute the corresponding linepair s2, t2:
			//mirror the positions at halfDim
			//TODO
			//TODO
			
			//[Paragraph after Eq. (17) in the paper]
			int twice_s1 = (2*s1) % dim[0];
			int twice_t1 = (2*t1) % dim[1];
		
			
			//Estimate FHat
			//4 special cases, where only a single line can be selected:
			//(0,0), (0, halfHeight), (halfWidth,0), (halfWidth, halfHeight)
			boolean specialCase = false;
			if( (s1 == 0 && t1 == 0 ) || 
				(s1 == halfDim[0] && t1 == 0) || 
				(s1 == 0 && t1 == halfDim[1]) || 
				(s1 == halfDim[0] && t1 == halfDim[1]))
			{
				System.out.println("Special Case");
				specialCase = true;
				//Eq. 15
				//FHat = N*(G(s,t)/W(0,0))
				//TODO compute FHatNext, use Complex class
				//TODO
				//TODO
				//TODO
				FHatNext.setRealAtIndex(s1, t1, (float) res.getReal());
				FHatNext.setImagAtIndex(s1, t1, (float) res.getImag());
			}
			else
			{
				//General case
				//Compute FHatNext for the general case Eq.9
				//TODO
				//TODO
				//TODO
				//TODO
				//TODO

				FHatNext.setRealAtIndex(s1, t1, (float) res_s1t1.getReal());
				FHatNext.setImagAtIndex(s1, t1, (float) res_s1t1.getImag());
				FHatNext.setRealAtIndex(s2, t2, (float) res_s2t2.getReal());
				FHatNext.setImagAtIndex(s2, t2, (float) res_s2t2.getImag());
			}
			
			//End iteration step by forming the new error spectrum
			updateErrorSpectrum(G, FHatNext, FHat, W, s1, t1, specialCase);
			
			//Get rid of rounding errors
			//G(t1,s1) and G(t2,s2) should be zero
			G.setAtIndex(s1, t1, 0);
			if(!specialCase)
			{
				G.setAtIndex(s2, t2, 0);
			}
			
			FHat = new Grid2DComplex(FHatNext);
			
			if(i % 100 == 0)
			{
				//For visualization, apply IFFT to the estimation
				Grid2DComplex FHatV = new Grid2DComplex(FHat);
				FHatV.transformInverse();
				Grid2D vis = new Grid2D(image);
				
				//Fill in the defect mask pixels with current estimation and remove the zero padding
				for(int x = 0; x < vis.getWidth(); x++)
				{
					for(int y = 0; y < vis.getHeight(); y++)
					{
						if(mask.getAtIndex(x, y) == 0)
						{
							vis.setAtIndex(x, y, FHatV.getRealAtIndex(x, y));
						}
					}
				}
				visualize.setSubGrid(i/100, vis);
				
			}
			
		}
		
		visualize.show();
		
		//Compute the inverse fourier transform of the estimated image
		FHat.transformInverse();
		
		//Fill in the defect mask pixels with the current estimation and remove the zero padding
		
		//TODO
		//TODO
		//TODO
		//TODO
		
		return result;
	}
	
	
	/**
	 * 
	 * Do the convolution of the m-times-n matrix F and W
	 * s,t is the position of the selected line pair, the convolution is simplified in the following way:
	 * G(k1,k2) = F(k1,k2) 'conv' W(k1,k2) 
	 *          = (F(s,t)W(k1-s,k2-t) + F*(s,t)W(k1+s,k2+t)) / (MN)
	 * where F* is the conjugate complex.
	 *
	 * @param G Fourier transformation of input image
	 * @param FHatNext currently estimated FT of the fixed image
	 * @param FHat previous estimated FT of the fixed image
	 * @param W Fourier transformation of the mask image
	 * @param s1 position of the selected line pair
	 * @param t1 position of the selected line pair
	 * @param specialCase 
	 */
	private static void updateErrorSpectrum(Grid2DComplex G, Grid2DComplex FHatNext, Grid2DComplex FHat, Grid2DComplex W, int s1, int t1, boolean specialCase) {
		int[] sz = FHatNext.getSize();
		
		// Accumulation: Update pair (s1,t1),(s2,t2)
		Complex F_st = new Complex(FHatNext.getRealAtIndex(s1, t1) - FHat.getRealAtIndex(s1, t1), FHatNext.getImagAtIndex(s1, t1) - FHat.getImagAtIndex(s1, t1));
		Complex F_st_conj = F_st.getConjugate();
		
		int MN = sz[0] * sz[1];
		
		// Compute the new error spectrum
		for(int j = 0; j < sz[1]; j++) 
		{
			for(int i = 0; i < sz[0]; i++) 
			{
				Complex GVal;
				if(specialCase) 
				{
					int xneg = (i - s1) % sz[0];
					int yneg = (j - t1) % sz[1];
					
					if(xneg < 0) 
					{
						xneg = sz[0] + xneg;
					}
					if(yneg < 0) 
					{
						yneg = sz[1] + yneg;
					}
					
					GVal = new Complex(G.getRealAtIndex(i, j), G.getImagAtIndex(i, j));
					Complex WNeg = new Complex(W.getRealAtIndex(xneg, yneg), W.getImagAtIndex(xneg, yneg));
					GVal.sub( ( F_st.mul(WNeg) ).div(MN) );
				
				}
				else
				{
					int xpos = (i + s1) % sz[0];
					int ypos = (j + t1) % sz[1];
					int xneg = (i - s1) % sz[0];
					int yneg = (j - t1) % sz[1];
					
					if(xneg < 0) 
					{
						xneg = sz[0] + xneg;
					}
					if(yneg < 0) 
					{
						yneg = sz[1] + yneg;
					}
					
					Complex WPos = new Complex(W.getRealAtIndex(xpos, ypos), W.getImagAtIndex(xpos, ypos));
					Complex WNeg = new Complex(W.getRealAtIndex(xneg, yneg), W.getImagAtIndex(xneg, yneg));
					GVal = new Complex(G.getRealAtIndex(i, j), G.getImagAtIndex(i, j));
					GVal = GVal.sub( ( ( F_st.mul(WNeg) ).add( F_st_conj.mul(WPos) ) ).div(MN) );
					

				}
				
				G.setRealAtIndex(i, j, (float) GVal.getReal());
				G.setImagAtIndex(i, j, (float) GVal.getImag());
			}
		}
	}
	
	public Grid2D interpolateMedian(Grid2D image, Grid2D defects, int kernelWidth, int kernelHeight)
	{
		//Pad the image. Otherwise, the filter will ignore kernelWidth/2 at each side of the image
		Grid2D paddedImage = new Grid2D(image.getWidth()+kernelWidth, image.getHeight()+kernelHeight);
		
		for(int i = 0; i <image.getWidth(); i++)
		{
			for(int j = 0; j < image.getHeight(); j++)
			{
				float val = image.getAtIndex(i, j);
				paddedImage.setAtIndex(i+kernelWidth/2, j+kernelHeight/2, val);
			}
		}
		paddedImage.show();
		
		MedianFilteringTool medFilt = new MedianFilteringTool();
		medFilt.configure(kernelWidth, kernelHeight);
				
		Grid2D medianFiltered = medFilt.applyToolToImage(paddedImage);
		
		Grid2D result = new Grid2D(image);
		
		for(int i = 0; i < image.getWidth(); i++)
		{
			for(int j = 0; j < image.getHeight(); j++)
			{
				if(defects.getAtIndex(i, j) == 0)
				{
					//medianFilteredImage is larger than original image
					result.setAtIndex(i, j, medianFiltered.getAtIndex(i+kernelWidth/2, j+kernelHeight/2));
				}
			}
		}
		
		return result;
		
	}
	

	public static void main(String[] args) {
		ImageJ ij =new ImageJ();
		DefectPixelInterpolation dpi = new DefectPixelInterpolation();
		
				
		//Load an image from file
		String filename = "D:/02_lectures/DMIP/exercises/2014/3/testimg.bmp";
		String filenameMask = "D:/02_lectures/DMIP/exercises/2014/3/mask.bmp";

		Grid2D image = ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
		image.show("Ideal Input Image");
		
		Grid2D mask = ImageUtil.wrapImagePlus(IJ.openImage(filenameMask)).getSubGrid(0);
		//Set some pixels as defect, elementwise multiply with defect pixel mask
		Grid2D defectImage = new Grid2D(image);
		//TODO
		defectImage.show("Defect Image");
		
		
		//Spatial Interpolation
		//Median Filter:
		int kernelWidth = 20;
		int kernelHeight = kernelWidth;
		Grid2D medianFiltered = dpi.interpolateMedian(defectImage, mask, kernelWidth, kernelHeight);
		medianFiltered.show("Median Filtered Image");
		
		//show difference image |Median - Original|
		Grid2D absDiffMedian = new Grid2D(image.getWidth(), image.getHeight());
		
		for(int i = 0; i < absDiffMedian.getWidth(); i++)
		{
			for(int j = 0; j < absDiffMedian.getHeight(); j++)
			{
				float val = Math.abs(medianFiltered.getAtIndex(i, j) - image.getAtIndex(i, j));
				absDiffMedian.setAtIndex(i, j, val);
			}
		}
		absDiffMedian.show("|Median - Original|");
		
		
		//Spectral Interpolation		
		boolean zeroPadding = true;
		int maxIter = 4000;
		
		//TODO
		spectralFiltered.show("Spectral Filtered Image");
		
		//show difference image |Spectral - Original|
		Grid2D absDiffSpectral = new Grid2D(image.getWidth(), image.getHeight());
		
		for(int i = 0; i < absDiffSpectral.getWidth(); i++)
		{
			for(int j = 0; j < absDiffSpectral.getHeight(); j++)
			{
				float val = Math.abs(spectralFiltered.getAtIndex(i, j) - image.getAtIndex(i, j));
				absDiffSpectral.setAtIndex(i, j, val);
			}
		}
		absDiffSpectral.show("|Spectral - Original|");

		//show difference between median and spectral
		Grid2D absDiffSpectralMedian = new Grid2D(image.getWidth(), image.getHeight());
		
		for(int i = 0; i < absDiffSpectralMedian.getWidth(); i++)
		{
			for(int j = 0; j < absDiffSpectralMedian.getHeight(); j++)
			{
				float val = Math.abs(spectralFiltered.getAtIndex(i, j) - medianFiltered.getAtIndex(i, j));
				absDiffSpectralMedian.setAtIndex(i, j, val);
			}
		}
		absDiffSpectralMedian.show("|Spectral - Median|");

	}

}

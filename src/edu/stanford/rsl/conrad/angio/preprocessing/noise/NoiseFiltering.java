/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.noise;

import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.filter.GaussianBlur;
import ij.process.FloatProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.BilateralFilteringTool;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class NoiseFiltering {

	
	
	public static Grid3D bilateralFilter(Grid3D noisy, int width, double sigDomain, double sigPhoto){
		int[] gSize = noisy.getSize();
		Grid3D filtered = (Grid3D)noisy.clone();
		BilateralFilteringTool bilat = new BilateralFilteringTool();
		if(width < 0){
			return filtered;
		}
		bilat.setParameters(width, sigDomain, sigPhoto);
		bilat.setConfigured(true);
		
		for(int k = 0; k < gSize[2]; k++){
			System.out.println("Bilateral filtering on "+String.valueOf(k+1)+" of "+String.valueOf(gSize[2]));
			try {
				Grid2D filt = bilat.applyToolToImage(filtered.getSubGrid(k));
				filtered.setSubGrid(k, filt);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return filtered;
	}
	
	public static Grid3D gaussianFilter(Grid3D noisy, double sigma){
		int[] gSize = noisy.getSize();
		ImagePlus imp = ImageUtil.wrapGrid3D(new Grid3D(noisy), "");
		ImageStack ims = imp.getStack();
		ImageStack imsFiltered = new ImageStack(gSize[0], gSize[1]);
		GaussianBlur gb = new GaussianBlur();
		
		for(int k = 0; k < gSize[2]; k++){
			System.out.println("Gaussian filtering on "+String.valueOf(k+1)+" of "+String.valueOf(gSize[2]));
			FloatProcessor fp = (FloatProcessor) ims.getProcessor(k+1);
			gb.blurFloat(fp, sigma, sigma, 0.01);
			imsFiltered.addSlice(fp);
		}
		ImagePlus impFiltered = new ImagePlus();
		impFiltered.setStack(imsFiltered);
		return ImageUtil.wrapImagePlus(impFiltered);
	}
	
	
	public static Grid2D gaussianFilter(Grid2D noisy, double sigma){
		FloatProcessor fp = ImageUtil.wrapGrid2D(new Grid2D(noisy));
		GaussianBlur gb = new GaussianBlur();
		
		System.out.println("Gaussian filtering.");
		gb.blurFloat(fp, sigma, sigma, 0.01);
		
		return ImageUtil.wrapImageProcessor(fp);
	}
}

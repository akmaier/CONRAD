/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

public class HysteresisThresholding {
	
	private double T1 = 0;
	private double T2 = 0;
	ImagePlus imp;

	public HysteresisThresholding(double tLow, double tHigh){
		this.T1 = tLow;
		this.T2 = tHigh;
	}

	/**
	 *  Main processing method for the Hysteresis_ object
	 *
	 *@param  imp
	 */
	public ImagePlus run(ImagePlus imp) {
		
		ImageStack stack = imp.getStack();
		ImageStack res_trin = new ImageStack(stack.getWidth(), stack.getHeight());
		ImageStack res_hyst = new ImageStack(stack.getWidth(), stack.getHeight());

		ImageProcessor tmp1;
		ImageProcessor tmp2;

		for (int s = 1; s <= stack.getSize(); s++) {
			System.out.println("Hysteresis Thresholding on slice "+s+" of "+stack.getSize());
			tmp1 = trin(stack.getProcessor(s), T1, T2);
			tmp2 = hyst(tmp1);
			res_trin.addSlice("", tmp1);
			res_hyst.addSlice("", tmp2);
		}
		//new ImagePlus("Trinarisation", res_trin).show();
		return new ImagePlus("Hysteresis", res_hyst);
	}

	/**
	 *  Main processing method for the Hysteresis_ object
	 *
	 *@param  impAsGrid
	 */
	public Grid3D run(Grid3D impAsGrid) {
		
		ImagePlus imp = ImageUtil.wrapGrid(impAsGrid, "");
		ImageStack stack = imp.getStack();
		ImageStack res_trin = new ImageStack(stack.getWidth(), stack.getHeight());
		ImageStack res_hyst = new ImageStack(stack.getWidth(), stack.getHeight());

		ImageProcessor tmp1;
		ImageProcessor tmp2;

		for (int s = 1; s <= stack.getSize(); s++) {
			System.out.println("Hysteresis Thresholding on slice "+s+" of "+stack.getSize());
			tmp1 = trin(stack.getProcessor(s), T1, T2);
			tmp2 = hyst(tmp1);
			res_trin.addSlice("", tmp1);
			res_hyst.addSlice("", tmp2);
		}
		//new ImagePlus("Trinarisation", res_trin).show();
		return ImageUtil.wrapImagePlus(new ImagePlus("Hysteresis", res_hyst));
	}
	
	/**
	 *  Main processing method for the Hysteresis_ object
	 *
	 *@param  impAsGrid
	 */
	public Grid2D run(Grid2D impAsGrid) {
		
		ImageProcessor imp = ImageUtil.wrapGrid2D((Grid2D)impAsGrid.clone());
		
		ImageProcessor tmp1;
		ImageProcessor tmp2;

		tmp1 = trin(imp, T1, T2);
		tmp2 = hyst(tmp1);
		
		//new ImagePlus("Trinarisation", res_trin).show();
		return ImageUtil.wrapImageProcessor(tmp2);
	}
	

	/**
	 *  Double thresholding
	 *
	 *@param  ima  original image
	 *@param  T1   high threshold
	 *@param  T2   low threshold
	 *@return      "trinarised" image
	 */
	ImageProcessor trin(ImageProcessor ima, double T1, double T2) {
		int la = ima.getWidth();
		int ha = ima.getHeight();
		ByteProcessor res = new ByteProcessor(la, ha);
		float pix;

		for (int x = 0; x < la; x++) {
			for (int y = 0; y < ha; y++) {
				pix = ima.getPixelValue(x, y);
				if (pix >= T1) {
					res.putPixel(x, y, 255);
				} else if (pix >= T2) {
					res.putPixel(x, y, 128);
				}
			}
		}
		return res;
	}


	/**
	 *  Hysteresis thresholding
	 *
	 *@param  ima  original image
	 *@return      thresholded image
	 */
	ImageProcessor hyst(ImageProcessor ima) {
		int la = ima.getWidth();
		int ha = ima.getHeight();
		ImageProcessor res = ima.duplicate();
		boolean change = true;

		// connection
		while (change) {
			change = false;
			for (int x = 1; x < la - 1; x++) {
				for (int y = 1; y < ha - 1; y++) {
					if (res.getPixelValue(x, y) == 255) {
						if (res.getPixelValue(x + 1, y) == 128) {
							change = true;
							res.putPixelValue(x + 1, y, 255);
						}
						if (res.getPixelValue(x - 1, y) == 128) {
							change = true;
							res.putPixelValue(x - 1, y, 255);
						}
						if (res.getPixelValue(x, y + 1) == 128) {
							change = true;
							res.putPixelValue(x, y + 1, 255);
						}
						if (res.getPixelValue(x, y - 1) == 128) {
							change = true;
							res.putPixelValue(x, y - 1, 255);
						}
						if (res.getPixelValue(x + 1, y + 1) == 128) {
							change = true;
							res.putPixelValue(x + 1, y + 1, 255);
						}
						if (res.getPixelValue(x - 1, y - 1) == 128) {
							change = true;
							res.putPixelValue(x - 1, y - 1, 255);
						}
						if (res.getPixelValue(x - 1, y + 1) == 128) {
							change = true;
							res.putPixelValue(x - 1, y + 1, 255);
						}
						if (res.getPixelValue(x + 1, y - 1) == 128) {
							change = true;
							res.putPixelValue(x + 1, y - 1, 255);
						}
					}
				}
			}
			if (change) {
				for (int x = la - 2; x > 0; x--) {
					for (int y = ha - 2; y > 0; y--) {
						if (res.getPixelValue(x, y) == 255) {
							if (res.getPixelValue(x + 1, y) == 128) {
								change = true;
								res.putPixelValue(x + 1, y, 255);
							}
							if (res.getPixelValue(x - 1, y) == 128) {
								change = true;
								res.putPixelValue(x - 1, y, 255);
							}
							if (res.getPixelValue(x, y + 1) == 128) {
								change = true;
								res.putPixelValue(x, y + 1, 255);
							}
							if (res.getPixelValue(x, y - 1) == 128) {
								change = true;
								res.putPixelValue(x, y - 1, 255);
							}
							if (res.getPixelValue(x + 1, y + 1) == 128) {
								change = true;
								res.putPixelValue(x + 1, y + 1, 255);
							}
							if (res.getPixelValue(x - 1, y - 1) == 128) {
								change = true;
								res.putPixelValue(x - 1, y - 1, 255);
							}
							if (res.getPixelValue(x - 1, y + 1) == 128) {
								change = true;
								res.putPixelValue(x - 1, y + 1, 255);
							}
							if (res.getPixelValue(x + 1, y - 1) == 128) {
								change = true;
								res.putPixelValue(x + 1, y - 1, 255);
							}
						}
					}
				}
			}
		}
		// suppression
		for (int x = 0; x < la; x++) {
			for (int y = 0; y < ha; y++) {
				if (res.getPixelValue(x, y) == 128) {
					res.putPixelValue(x, y, 0);
				}
			}
		}
		return res;
	}
	
}

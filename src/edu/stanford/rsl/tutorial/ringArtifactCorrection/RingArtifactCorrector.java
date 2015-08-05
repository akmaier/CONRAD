package edu.stanford.rsl.tutorial.ringArtifactCorrection;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.MeanFilteringTool;
import edu.stanford.rsl.conrad.filtering.MedianFilteringTool;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class RingArtifactCorrector {
	private int radialFilterWidth, azimuthFilterWidth; //filter widths
	private boolean showSteps = false; //show images for each step
	
	public RingArtifactCorrector(int radialFilterWidth, int azimuthFilterWidth) {
		this.radialFilterWidth = radialFilterWidth;
		this.azimuthFilterWidth = azimuthFilterWidth;
	}
	
	public RingArtifactCorrector() {
		this.radialFilterWidth = 40; //suggest parameter for example image
		this.azimuthFilterWidth = 40;
	}
	
	public Grid2D getRingArtifactCorrectedImage(Grid2D input) {
		ImagePlus mask = getMask(input);
		Grid2D thresholded = thresholdImage(input,mask);
		PolarConverter conv = new PolarConverter();
		Grid2D polarConverted = conv.convertToPolar(thresholded);
		Grid2D radialFilt = radialMedianFilter((Grid2D) polarConverted.clone(),radialFilterWidth);
		Grid2D diff = subtractImage((Grid2D) polarConverted.clone(),radialFilt);
		Grid2D aziFilt = azimuthalMedianFilter((Grid2D) diff.clone(), azimuthFilterWidth);
		Grid2D artifactsCart = conv.convertToCartesian((Grid2D) aziFilt.clone());
		Grid2D result = subtractImage(input,artifactsCart);
		ImagePlus res = ImageUtil.wrapGrid(result, "RingArtifact Corrected");
		res.getProcessor().resetMinAndMax();
		result = ImageUtil.wrapImageProcessor(res.getProcessor());
		result = mathMin(result);
		if (showSteps) {
			mask.show("Threshold mask");
			thresholded.show("Thresholded");
			polarConverted.show("Converted to polar coordinates");
			radialFilt.show("Radial median filtered");
			diff.show("Difference image: radial-median & unfiltered");
			aziFilt.show("Azimuthal mean filtered difference");

		}
		return result;
	}
	
	public void setRadialFilterWidth(int width) {
		this.radialFilterWidth = width;
	}
	
	public void setAzimuthalFilterWidth (int width) {
		this.azimuthFilterWidth = width;
	}
	
	public void setShowSteps(boolean show) {
		this.showSteps = show;
	}
	
	private static Grid2D mathMin(Grid2D slice) {
		Grid2D res = (Grid2D) slice.clone();
		for (int x=0; x<res.getWidth(); x++) {
			for (int y=0; y<res.getHeight(); y++) {
				if (res.getPixelValue(x, y) < 0) { //set values below 0 to 0
					res.putPixelValue(x, y, 0.0f);
				}
			}
		}
		return res;
	}
	
	private static ImagePlus getMask(Grid2D slice) { //automatic object segmentation
		IJ.showStatus("Thresholding...");
		ImagePlus imp = ImageUtil.wrapGrid(slice, "mask");
		ImageProcessor proc = imp.getProcessor();
		proc = proc.convertToByte(true);
		proc.autoThreshold();
		proc.dilate();
		int nErode = 8; //empirically chosen value
		for (int i=0; i<nErode; i++) {
			proc.erode();
		}
		imp.setProcessor(proc);
		return imp;
	}
	
	private static Grid2D thresholdImage(Grid2D slice, ImagePlus mask) {
		Grid2D res = (Grid2D) slice.clone();
		for (int x=0; x<res.getWidth(); x++) {
			for (int y=0; y<res.getHeight(); y++) {
				if (mask.getPixel(x, y)[0] == 0) {
					res.putPixelValue(x, y, 0.0f);
				}
			}
		}
		return res;
	}
	
	private static Grid2D radialMedianFilter(Grid2D polar, int filterW) {
		IJ.showStatus("Radial median filtering...");
		MedianFilteringTool medfilt = new MedianFilteringTool();
		Grid2D[] subGrids = radialFilterAreas(polar); //split image into 3 regions
		medfilt.setKernelHeight(1);
		medfilt.setKernelWidth(filterW/3); //filter sub images with corresponding filter widths
		Grid2D inner = medfilt.getMedianFilteredImage(subGrids[0]);
		medfilt.setKernelWidth(2*filterW/3);
		Grid2D middle = medfilt.getMedianFilteredImage(subGrids[1]);
		medfilt.setKernelWidth(filterW);
		Grid2D outer = medfilt.getMedianFilteredImage(subGrids[2]);
		Grid2D[] filteredSubGrids = {inner,middle,outer};
		Grid2D merged = mergeRadialAreas(filteredSubGrids); //merge filtered images 
		return merged;
	}
	
	private static Grid2D azimuthalMedianFilter(Grid2D polar, int filterW) {
		IJ.showStatus("Azimuthal low-pass filtering...");
		Grid2D[] subGrids = radialFilterAreas(polar);
		MeanFilteringTool meanfilt = new MeanFilteringTool();
		if (filterW/3 % 2 == 0) { //meanfilter needs odd filter width
			meanfilt.configure(1, (filterW/3)+1);
		} else {
			meanfilt.configure(1, filterW/3);
		}
		Grid2D inner = meanfilt.applyToolToImage(subGrids[0]);
		if (2*filterW/3 % 2 == 0) {
			meanfilt.configure(1, (2*filterW/3)+1);
		} else {
			meanfilt.configure(1, 2*filterW/3);
		}
		Grid2D middle = meanfilt.applyToolToImage(subGrids[1]);
		if (filterW % 2 == 0) {
			meanfilt.configure(1, filterW+1);
		} else {
			meanfilt.configure(1, filterW);
		}
		Grid2D outer = meanfilt.applyToolToImage(subGrids[2]);
		Grid2D[] filteredSubGrids = {inner,middle,outer};
		Grid2D merged = mergeRadialAreas(filteredSubGrids);
		
		
		return merged;
	}
	
	private static Grid2D mergeRadialAreas(Grid2D[] subGrids) {
		int subWidth0 = subGrids[0].getWidth();
		int subWidth1 = subGrids[1].getWidth();
		int subWidth2 = subGrids[2].getWidth();
		int width = subWidth0 + subWidth1 + subWidth2;
		Grid2D merged = new Grid2D(width,subGrids[0].getHeight());
		for (int x=0; x<width; x++) {
			for (int y=0; y<subGrids[0].getHeight(); y++) {
				double val = 0.0;
				if (x < subWidth0) {
					val = subGrids[0].getPixelValue(x, y);
				} else if (x >= subWidth0 && x < subWidth0 + subWidth1) {
					val = subGrids[1].getPixelValue(x-subWidth0,y);
				} else {
					val = subGrids[2].getPixelValue(x-(subWidth0+subWidth1), y);
				}
				merged.putPixelValue(x, y, val);
			}
		}
		return merged;
	}

	
	
	private static Grid2D[] radialFilterAreas(Grid2D polar) {
		int[] areas = radialFilterLimits(polar);
		Grid2D inner = new Grid2D(areas[0],polar.getHeight());
		Grid2D middle = new Grid2D(areas[0],polar.getHeight());
		Grid2D outer = new Grid2D(polar.getWidth()-areas[1],polar.getHeight());
		for (int x=0; x<areas[0]; x++) {
			for (int y=0; y<polar.getHeight(); y++) {
				double innerVal = polar.getAtIndex(x, y);
				double middleVal = polar.getAtIndex(x+areas[0], y);
				double outerVal = polar.getAtIndex(x+areas[1], y);
				inner.putPixelValue(x, y, innerVal);
				middle.putPixelValue(x, y, middleVal);
				outer.putPixelValue(x, y, outerVal);
			}
		}
		Grid2D[] subGrids = {inner,middle,outer};
		return subGrids;
	}
	
	
	private static int[] radialFilterLimits(Grid2D polar) {
		int[] areas = new int[3];
		int width = polar.getWidth();
		areas[0] = (int)((width+1)/3.0);
		areas[1] = (int)((2*width+1)/3.0);
		areas[2] = width;
		return areas;
	}
	
	private static Grid2D subtractImage(Grid2D polarInput, Grid2D artifacts) {
		IJ.showStatus("Getting difference image...");
		Grid2D res = (Grid2D) polarInput.clone();
		int width = polarInput.getWidth();
		int height = polarInput.getHeight();
		for (int x=0; x<width; x++) {
			for (int y=0; y<height; y++) {
				float diff = polarInput.getPixelValue(x, y) - artifacts.getPixelValue(x, y);
				res.putPixelValue(x, y, diff);
			}
		}
		return res;
	}

	Grid3D loadReconstruction(String path) {
		Grid3D impAsGrid = null;
		try {
			ImagePlus imp = IJ.openImage(path);
			impAsGrid = ImageUtil.wrapImagePlus(imp);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return impAsGrid;
	}

}

/*
 * Copyright (C) 2010-2015 Florian Gabsteiger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.image;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.gui.Roi;

public class HistogramPercentile {

	private Roi roi = null;
	
	private Grid2D img;
	
	private int numBins = 255;
	
	private long[] hist;
	private boolean minSet = false;
	private float min;
	private boolean maxSet = false;
	private float max;
	private float delta;
	private int size;
	
	private boolean initialized = false;
	
	
	public static void main(String[] args) {
		Grid2D img = ImageUtil.wrapImageProcessor(IJ.openImage(".../sinogram.tif").getProcessor());
		HistogramPercentile perc = new HistogramPercentile(img);
		float p = perc.getPercentile(0.5);
		System.out.println(p);
	}
	
	public HistogramPercentile(Grid2D img){
		this.img = img;
	}
	
	public float getPercentile(double perc){
		return getPercentile(perc, false);
	}

	public float getPercentile(double perc, boolean ignoreFirst){
		if(!initialized){
			initialize();
		}
		int startIdx = (ignoreFirst)?1:0;
		double numToExceed = perc*(size - startIdx*hist[0]);
		int currentNum = 0;
		for(int i = startIdx; i < numBins; i++){
			currentNum += hist[i];
			if(currentNum >= numToExceed){
				return (i*delta+min);
			}
		}
		return ((numBins-1)*delta+min);
	}
	
	private void initialize() {
		float min = +Float.MAX_VALUE;
		float max = -Float.MAX_VALUE;
		int size = 0;
		int[] gSize = img.getSize();
		if(this.roi == null){
			this.roi = new Roi(0,0,gSize[0]+1,gSize[1]+1);
		}
		for(int i = 0; i < gSize[0]; i++){
			for(int j = 0; j < gSize[1]; j++){
				if(roi.contains(i, j)){
					float val = img.getAtIndex(i, j);
					min = Math.min(min,val);
					max = Math.max(max,val);
				}
			}
		}
		if(!minSet){
			this.min = min;
		}
		if(!maxSet){
			this.max = max;
		}
		this.delta = (this.max-this.min)/(numBins-1);
		this.hist = new long[numBins];
		for(int i = 0; i < gSize[0]; i++){
			for(int j = 0; j < gSize[1]; j++){
				if(roi.contains(i, j)){
					float val = img.getAtIndex(i, j);
					int bin = (int)((val-this.min)/delta);
					if(bin >= 0 && bin < numBins){
						hist[bin] += 1;
						size++;
					}
				}
			}
		}
		this.size = size;
		initialized = true;
	}

	public long[] getHist() {
		return hist;
	}

	public int getNumBins() {
		return numBins;
	}

	public void setNumBins(int numBins) {
		this.numBins = numBins;
		initialized = false;
	}
	
	public Roi getRoi() {
		return roi;
	}

	public void setRoi(Roi roi) {
		this.roi = roi;
		this.initialized = false;
	}
	
	public void setMin(float min) {
		this.min = min;
		this.minSet = true;
	}
	
	public void setMax(float max) {
		this.max = max;
		this.maxSet = true;
	}
}

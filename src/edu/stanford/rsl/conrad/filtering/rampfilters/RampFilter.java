package edu.stanford.rsl.conrad.filtering.rampfilters;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.io.SafeSerializable;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

/**
 * Class to create Ramp Filters as described in Kak and Slaney 1988 (pp. 72) 
 * This class is the blue print for an arbitrary Ram Lak Filter. Subclasses implement different filterings.
 * 
 * @see "RamLakRampFilter for an implementation without filtering"
 * 
 * @see "SheppLoganRampFilter, CosineRampFilter, HanningRampFilter, HammingRampFilter for different implementations of filters"
 * 
 * @author Andreas Maier
 *
 */
public abstract class RampFilter implements Cloneable, SafeSerializable, GUIConfigurable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -6709816992630498882L;

	protected boolean debug = false;
	protected boolean configured;

	protected double physicalPixelWidthInMilimeters = 1;

	protected double cutOffFrequency = 1;

	protected double sourceToDetectorDistance = 1;
	protected double [] filter = null;

	protected int currentWidth = -1;
	private FloatProcessor currentProcessor = null;


	public void setConfiguration(Configuration config){
		this.setCutOffFrequency(config.getCutOffFrequency());
		this.setPhysicalPixelWidthInMilimeters(config.getGeometry().getPixelDimensionX());
		this.setSourceToAxisDistance(config.getGeometry().getSourceToAxisDistance());
		this.setSourceToDetectorDistance(config.getGeometry().getSourceToDetectorDistance());
	}

	public double getSourceToDetectorDistance() {
		return sourceToDetectorDistance;
	}

	public void setSourceToDetectorDistance(double sourceToDetectorDistance) {
		this.sourceToDetectorDistance = sourceToDetectorDistance;
	}

	protected double sourceToAxisDistance = 1;

	public double getSourceToAxisDistance() {
		return sourceToAxisDistance;
	}

	public void setSourceToAxisDistance(
			double sourceToAxisDistance) {
		this.sourceToAxisDistance = sourceToAxisDistance;
	}

	public double getCutOffFrequency() {
		return cutOffFrequency;
	}

	public void setCutOffFrequency(double cutOffFrequency) {
		this.cutOffFrequency = cutOffFrequency;
	}

	public double getPhysicalPixelWidthInMilimeters() {
		return physicalPixelWidthInMilimeters;
	}

	public void setPhysicalPixelWidthInMilimeters(double detectorWidth) {
		this.physicalPixelWidthInMilimeters = detectorWidth;
	}

	public abstract String getRampName();

	/**
	 * Returns the filter for one detector row as complex double array (JTransforms format)
	 * @param width the width of the detector row
	 * @return the filter in Fourier domain
	 */
	public double [] getRampFilter1D(int width){
		if (width != currentWidth) filter = null;
		if (filter == null) {
			// Create a suitable filter in Fourier space
			double scaling = 1;//sourceToAxisDistance * physicalPixelWidthInMilimeters / sourceToDetectorDistance;
			double [] filter = getFilterInFourierSpace(width);
			// Apply the filter until the center point
			for (int i = 0; i < width/2+1; i++){
				double ku = 2 * Math.PI * i / width;
				// Compute Weight
				ku = getFilterWeight(ku);
				// Apply to complex value
				filter[2 * i] *= ku * scaling;
				filter[(2 * i)+1] *= ku * scaling;
			}
			// Force Symmetry
			DoubleArrayUtil.forceSymmetryComplexDoubleArray(filter);
			currentWidth = width;
			this.filter = filter;
		}
		return filter;
	}

	/**
	 * Creates an ImagePlus to display the filter
	 * @param width width of the image
	 * @return the ImagePlus
	 */
	public ImagePlus getImagePlusFromRampFilter(int width){
		ImagePlus image = new ImagePlus();
		width = FFTUtil.getNextPowerOfTwo(width);
		FloatProcessor filterProcessor = getRampFilter(width);
		ImageStack stack = new ImageStack(width, width);
		stack.addSlice(this.getRampName(), filterProcessor);
		image.setStack(this.getRampName(), stack);
		return image;
	}


	/**
	 * Method to generate a filter for in Fourier space given the width of the image to filter
	 * @param width the width of the input image
	 * @return the filter as ImageProcessor
	 */
	public FloatProcessor getRampFilter(int width) {
		width = FFTUtil.getNextPowerOfTwo(width * 2);
		if (width != currentWidth) { // the ramp filter has to be created only once, if used for batch processing.
			double [] filter = getRampFilter1D(width);
			currentProcessor = new FloatProcessor(width,width);
			for (int i = 0; i < width; i++){
				for (int j = 0; j < width; j++){
					currentProcessor.putPixelValue(i, j, FFTUtil.abs(i, filter));
				}
			}
		}
		return currentProcessor;
	}

	public abstract double getFilterWeight(double ku);

	public abstract RampFilter clone();

	/**
	 * Creates a Ram Lak filter in Fourier space after Kak and Slaney (pp. 91)
	 * @param width the width of the filter
	 * @return an array of complex values in JTransforms format, i.e. real followed by complex part and so on ...
	 */
	protected double [] getFilterInFourierSpace(int width){
		double [] filter = new double[width * 2];
		// Negative comb
		for (int i = 0; i < width; i++){
			int index = (-width/2) + 1 + i;
			if (debug) {
				System.out.println(index);
				System.out.println(Math.abs(index % 2));
			}
			if (Math.abs(index % 2) == 1) filter[i*2] = -1 / (2*Math.pow(index * Math.PI * physicalPixelWidthInMilimeters, 2));
		}
		// Peak in the center of the filter
		filter[width-2] = 1 / (8 * Math.pow(physicalPixelWidthInMilimeters,2));
		if (debug) FFTUtil.printComplex(filter);
		// Prepare for FFT
		DoubleFFT_1D fft = new DoubleFFT_1D(width);
		// FFT
		fft.complexForward(filter);
		return filter;
	}

	public String toString(){
		return this.getRampName();
	}

	public static RampFilter[] getAvailableRamps(){
		RampFilter [] filters = {new RamLakRampFilter(), new SheppLoganRampFilter(), new SheppLoganRampFilterWithRollOff(), new CosineRampFilter(), new HammingRampFilter(), new HanningRampFilter(), new ArbitraryRampFilter()};
		return filters;
	}
	
	public boolean isConfigured(){
		return configured;
	}
	
	public void configure() throws Exception{
		
	}

	public void prepareForSerialization(){
		
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
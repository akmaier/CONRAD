package edu.stanford.rsl.conrad.filtering;


import java.awt.Rectangle;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.plugin.filter.GaussianBlur;
import ij.process.FloatProcessor;

public class DynamicDensityOptimizationScatterCorrectionTool extends
IndividualImageFilteringTool {


	/**
	 * 
	 */
	private static final long serialVersionUID = -8866437142171005094L;
	/**
	 * 
	 */

	private double sigma = 175;
	private double weight = 0.99;

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
	}

	/**
	 * Dynamic Density Optimization of a float image. 'fp' must have a valid snapshot. 
	 * Almost identical to sharpenFloat in UnsharpMask.java from ImageJ.
	 * Unsharp mask is added instead of subtracted as the filter is applied in log domain.
	 * 
	 * @param fp the Processor to be masked
	 * @param sigma the standard deviation
	 * @param weight the weight of the mask [0,1]
	 */
	public void sharpenFloat(FloatProcessor fp, double sigma, float weight) {
		GaussianBlur gb = new GaussianBlur();
		gb.blurGaussian(fp, sigma, sigma, 0.01);
		if (Thread.currentThread().isInterrupted()) return;
		float[] pixels = (float[])fp.getPixels();
		float[] snapshotPixels = (float[])fp.getSnapshotPixels();
		int width = fp.getWidth();
		Rectangle roi = fp.getRoi();
		for (int y=roi.y; y<roi.y+roi.height; y++)
			for (int x=roi.x, p=width*y+x; x<roi.x+roi.width; x++,p++)
				pixels[p] = (snapshotPixels[p] + weight*pixels[p])/(1f - weight);
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		FloatProcessor fl = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		fl.setPixels(imageProcessor.getBuffer());
		fl.snapshot();
		sharpenFloat(fl, sigma, (float) weight);
		return imageProcessor;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		DynamicDensityOptimizationScatterCorrectionTool clone = new DynamicDensityOptimizationScatterCorrectionTool();
		clone.sigma = sigma;
		clone.weight = weight;
		clone.setConfigured(configured);
		return clone;
	}

	@Override
	public String getToolName() {
		return "Dynamic Density Optimization Scatter Correction";
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@article{Dragusin08-AIO,"+
		"author={O Dragusin and H Bosmans and C Pappas and W Desmet},"+
		"title={An investigation of flat panel equipment variables on image quality with a dedicated cardiac phantom},"+
		"journal={Physics in Medicine and Biology},"+
		"volume={53},"+
		"number={18},"+
		"pages={4927-4940},"+
		"url={http://stacks.iop.org/0031-9155/53/i=18/a=005},"+
		"year={2008}"+
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "O Dragusin, H Bosmans, C Pappas, W Desmet." +
		"An investigation of flat panel equipment variables on image quality with a dedicated cardiac phantom." +
		"Phys. Med. Biol. 53:4927-40. 2008";
	}

	@Override
	public void configure() throws Exception {
		sigma = UserUtil.queryDouble("Enter Sigma", sigma);
		weight = UserUtil.queryDouble("Enter Weight", weight);
		configured = true;
	}

	/**
	 * Scatter correction is device dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	/**
	 * @return the sigma
	 */
	public double getSigma() {
		return sigma;
	}

	/**
	 * @param sigma the sigma to set
	 */
	public void setSigma(double sigma) {
		this.sigma = sigma;
	}

	/**
	 * @return the weight
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * @param weight the weight to set
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

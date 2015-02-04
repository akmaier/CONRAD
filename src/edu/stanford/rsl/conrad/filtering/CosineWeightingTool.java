package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.Configuration;


/**
 * Filtering tool to apply cosine weights. Weights can either be computed from six geometry parameters or loaded
 * from an ImagePlus. Raw cosine weights are sometimes provided as calibration data from some manufacturers.
 * 
 * Computation from the geometry parameters follows Kak & Slaney, Principles of Computerized Tomographic Imaging, 1988;
 * 
 * @author Andreas Maier
 *
 */
public class CosineWeightingTool extends IndividualImageFilteringTool {

	private static final long serialVersionUID = -4985677804638655239L;

	double [] [] cosineWeights = null;

	int detectorWidth = 0;
	int detectorHeight = 0;

	double pixelDimensionX = 0;
	double pixelDimensionY = 0;

	double sourceToDetectorDistance = 0;
	double sourceToAxisDistance = 0;

	private boolean weightsAvailable = false;


	/**
	 * Computes the cosine weights as described in Kak & Slaney, Computerized Tomographic Imaging, 1988
	 * @throws Exception if not all parameters are set.
	 */
	public synchronized void generateCosineWeights() throws Exception{
		if ((sourceToDetectorDistance == 0)
				|| (sourceToAxisDistance == 0)
				|| (detectorWidth == 0)
				|| (detectorHeight == 0)
				|| (pixelDimensionX == 0)
				|| (pixelDimensionY == 0)) {
			throw new Exception("Please configure sourceToDetectorDistance, sourceToAxisDistance, pixelDimensionX, pixelDimensionY, detectorWidth, and detectorHeight befor calling this method.");
		} else { // all set; compute weights
			//	Compute center point of detector 

			double detPiercingPointX = (((double) detectorWidth) / 2) * pixelDimensionX;
			double detPiercingPointY = (((double) detectorHeight) / 2) * pixelDimensionY;
			this.cosineWeights = new double[detectorWidth][detectorHeight];
			for (int i=0;i<detectorWidth;i++){
				// Cone weight as seen in step 1 of Kak & Slaney
				// transform to virtual detector centered at (uPrime = 0, vPrime = 0);
				// i.e.  u and v are transformed to a virtual detector centered at (0,0) (by a scale factor of SAD/SID),
				//       then cosineWeight = SAD / sqrt(SAD^2 + u'^2 + v'^2), which has the familiar form in many books.
				double uPrimeSquare = Math.pow(((((double)i) + 0.5)*pixelDimensionX) - detPiercingPointX, 2);
				for (int j = 0; j< detectorHeight; j++){
					//System.out.println("doing " + i + " " + j + " of " + detectorWidth + " " + detectorHeight);
					double vPrimeSquare = Math.pow(((((double) j)+0.5)*pixelDimensionY) - detPiercingPointY, 2);
					this.cosineWeights[i][j] = this.sourceToDetectorDistance / Math.sqrt( Math.pow(sourceToDetectorDistance, 2) + uPrimeSquare + vPrimeSquare);
				}
			}

		}
	}

	public double[][] getCosineWeights() {
		return cosineWeights;
	}

	public void setCosineWeights(double[][] cosineWeights) {
		this.cosineWeights = cosineWeights;
	}

	public double getPixelDimensionX() {
		return pixelDimensionX;
	}

	public void setPixelDimensionX(double pixelDimensionX) {
		this.pixelDimensionX = pixelDimensionX;
	}

	public double getPixelDimensionY() {
		return pixelDimensionY;
	}

	public void setPixelDimensionY(double pixelDimensionY) {
		this.pixelDimensionY = pixelDimensionY;
	}

	public double getSourceToDetectorDistance() {
		return sourceToDetectorDistance;
	}

	public void setSourceToDetectorDistance(double sourceToDetectorDistance) {
		this.sourceToDetectorDistance = sourceToDetectorDistance;
	}

	public int getDetectorWidth() {
		return detectorWidth;
	}

	public void setDetectorWidth(int detectorWidth) {
		this.detectorWidth = detectorWidth;
	}

	public int getDetectorHeight() {
		return detectorHeight;
	}

	public void setDetectorHeight(int detectorHeight) {
		this.detectorHeight = detectorHeight;
	}

	public double getSourceToAxisDistance() {
		return sourceToAxisDistance;
	}

	public void setSourceToAxisDistance(
			double sourceToAxisDistance) {
		this.sourceToAxisDistance = sourceToAxisDistance;
	}

	public void setConfiguration(Configuration config){
		this.setDetectorHeight(config.getGeometry().getDetectorHeight());
		this.setDetectorWidth(config.getGeometry().getDetectorWidth());
		this.setPixelDimensionX(config.getGeometry().getPixelDimensionX());
		this.setPixelDimensionY(config.getGeometry().getPixelDimensionY());
		this.setSourceToAxisDistance(config.getGeometry().getSourceToAxisDistance());
		this.setSourceToDetectorDistance(config.getGeometry().getSourceToDetectorDistance());
	}

	@Override
	public IndividualImageFilteringTool clone() {
		CosineWeightingTool filter = new CosineWeightingTool();
		filter.setDetectorHeight(getDetectorHeight());
		filter.setDetectorWidth(getDetectorWidth());
		filter.setPixelDimensionX(getPixelDimensionX());
		filter.setPixelDimensionY(getPixelDimensionY());
		filter.setSourceToAxisDistance(getSourceToAxisDistance());
		filter.setSourceToDetectorDistance(getSourceToDetectorDistance());
		filter.setCosineWeights(this.cosineWeights);
		filter.configured = configured;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Cosine Weighting Filter";
	}


	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		if (!weightsAvailable) {
			try {
				this.generateCosineWeights();
				weightsAvailable = true;
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		//System.out.println("doing slice");
		if (imageProcessor.getWidth() > detectorWidth){
			throw new Exception("Detector dimension does not fit projection data.");
		}
		int offset = 0;
		if (imageProcessor.getWidth() < detectorWidth){
			offset = (detectorWidth - imageProcessor.getWidth()) / 2;
		}
		for (int k = 0; k < imageProcessor.getWidth(); k++){
			for (int j = 0; j < imageProcessor.getHeight(); j++){
				//System.out.println("doing " + k + " " + j + " of " + imageProcessor.getWidth() + " " + imageProcessor.getHeight());
				double value = imageProcessor.getPixelValue(k, j) * this.cosineWeights[k+offset][j];
				//System.out.println("putting " + value);
				imageProcessor.putPixelValue(k, j, value);
				//System.out.println("put.");
			}
			//System.out.println("done j in " + k);
		}
		//System.out.println("done");
		return imageProcessor;
	}


	public void prepareForSerialization(){
		super.prepareForSerialization();
		cosineWeights = null;
		weightsAvailable = false;
	}


	@Override
	public void configure() {
		setConfiguration(Configuration.getGlobalConfiguration());
		try {
			generateCosineWeights();
			setConfigured(true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@BOOK{Kak88-POC,\n" +
				"  author = {{Kak}, A. C. and {Slaney}, M.},\n" +
				"  title = {{Principles of Computerized Tomographic Imaging}},\n" +
				"  publisher = {IEEE Service Center},\n" +
				"  address = {Piscataway, NJ, United States},\n" +
				"  year = {1988}\n" +
				"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Kak AC, Slaney M, Principles of Computerized Tomographic Imaging, IEEE Service Center, Piscataway, NJ, United States 1988.";
	}

	/**
	 * Cosine filtering depends on the projection geometry and is hence not device depdendent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
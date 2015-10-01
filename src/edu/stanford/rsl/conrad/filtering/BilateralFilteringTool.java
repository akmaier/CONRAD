package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class BilateralFilteringTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8308762121723928775L;
	private double sigma_d = 2.0;
	private double sigma_r = 0.001;
	private int width = 5;

	@Override
	public IndividualImageFilteringTool clone() {
		BilateralFilteringTool clone = new BilateralFilteringTool();
		clone.sigma_d = sigma_d;
		clone.sigma_r = sigma_r;
		clone.width = width;
		clone.configured = configured;
		return clone;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor)
	throws Exception {
		Grid2D filtered = new Grid2D(imageProcessor);
		for (int i = 0; i< filtered.getWidth(); i++){
			for (int j = 0; j < filtered.getHeight(); j++){
				double value = computeKernel(imageProcessor, i, j);
				filtered.putPixelValue(i, j, value);
			}
		}
		return filtered;
	}

	@Override
	public String getToolName() {
		return "Bilateral Filtering Tool";
	}



	private double computeGeometricCloseness(int i, int j, int x, int y){
		return Math.exp(- 0.5 * Math.pow((computeEuclidianDistance(i,j,x,y) / sigma_d), 2));
	}

	private double computeEuclidianDistance(int i, int j, int x, int y){
		return Math.sqrt(Math.pow(i-x ,2) + Math.pow((j-y),2));
	}

	private double computeIntensityDistance(Grid2D imageProcessor, int i, int j, int x, int y){
		double x_val = imageProcessor.getPixelValue(x, y);
		double xi_val = imageProcessor.getPixelValue(i, j);
		return Math.abs(x_val - xi_val);
	}

	private double computePhotometricDistance(Grid2D imageProcessor, int i, int j, int x, int y){
		return Math.exp(-0.5 * Math.pow(computeIntensityDistance(imageProcessor, i ,j,x,y) / sigma_r, 2));
	}

	private double computeKernel(Grid2D imageProcessor, int x, int y){
		double sumWeight = 0;
		double sumFilter = 0;
		// No filtering at the image boudaries;
		if ((x < (width/2)) || (x+(width/2)+1 >= imageProcessor.getWidth())
				|| (y < (width/2)) || (y+(width/2)+1 >= imageProcessor.getHeight())){
			sumWeight = 1;
			sumFilter = imageProcessor.getPixelValue(x, y);
		} else {
			for (int i = x-(width/2); i < x+(width/2)+1; i++){
				for (int j = y-(width/2); j < y+(width/2)+1; j++){
					double currentWeight = computePhotometricDistance(imageProcessor, i, j, x, y) * computeGeometricCloseness(i,j,x,y);
					//System.out.println("Photo " + computePhotometricDistance(imageProcessor, i, j, x, y));
					//System.out.println("Geom " + computeGeometricCloseness(i, j, x, y));
					sumWeight += currentWeight;
					sumFilter += currentWeight * imageProcessor.getPixelValue(i, j);
				}
			}
		}
		return sumFilter / sumWeight;
	}


	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Tomasi98-BFF,\n" +
		"  author = {Tomasi, C. and Manduchi, R.},\n" +
		"  title = {Bilateral Filtering for Gray and Color Images},\n" +
		"  booktitle = {ICCV '98: Proceedings of the Sixth International Conference on Computer Vision},\n" +
		"  year = {1998},\n" +
		"  isbn = {81-7319-221-9},\n" +
		"  pages = {839-846},\n" +
		"  publisher = {IEEE Computer Society},\n" +
		"  address = {Washington, DC, USA},\n" +
		"}\n";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Tomasi C, Maduchi R, Bilateral Filtering for Gray and Color Images. In: ICCV '98: Proceedings of the Sixth International Conference on Computer Vision, pp. 839-846, IEEE Computer Society, Washington, DC, United States 1998.";
	}

	@Override
	public void configure() throws Exception {
		width = UserUtil.queryInt("Enter Width", width);
		sigma_r = UserUtil.queryDouble("Sigma for photometric distance", sigma_r);
		sigma_d = UserUtil.queryDouble("Sigma for geometric distance", sigma_d);
		configured = true;
	}

	/**
	 * Bilateral Filtering is just just for noise filtering here and is hence not device dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	/**
	 * Allows changing of filter parameters without the need for calling the User-Query dialog. 
	 * @param width The kernel width
	 * @param domainSigma The domain standard deviation used for spatial filtering
	 * @param photometricSigma The photometric standard deviation
	 */
	public void setParameters(int width, double domainSigma, double photometricSigma){
		this.width = width;
		this.sigma_d = domainSigma;
		this.sigma_r = photometricSigma;
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

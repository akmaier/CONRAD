package edu.stanford.rsl.conrad.filtering.redundancy;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;


public class WesargWeightingTool extends ParkerWeightingTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2453380390532119284L;

	
	private double deltaX = 0;
	protected boolean filter = false;
	private double percent = 0.25;

	@Override
	protected double computeParkerWeight(int x, double beta){
		double value = (this.pixelDimensionX * (x - (detectorWidth/2)));
		double alpha = Math.atan(value / sourceToDetectorDistance);
		return computeWesargWeight(alpha, beta);
	}
	
	@Override
	public IndividualImageFilteringTool clone() {
		WesargWeightingTool clone = new WesargWeightingTool();
		clone.setDetectorWidth(this.getDetectorWidth());
		clone.setPixelDimensionX(this.getPixelDimensionX());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setNumberOfProjections(numberOfProjections);
		clone.setPrimaryAngles(getPrimaryAngles());
		clone.configured = configured;
		clone.offset = offset;
		clone.deltaX = deltaX;
		clone.percent = percent;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Wesarg Redundancy Weighting Filter";
	}

	public double [] computeParkerWeights1D(int projNum){
		checkDelta();
		double beta = projNum;
		double [] minmax = null;
		double maxRange = (Math.PI + (2 * delta));
		if (debug) System.out.println("Angular Max Range: " + maxRange * 180 / Math.PI);
		if (getPrimaryAngles() != null){
			minmax = DoubleArrayUtil.minAndMaxOfArray(getPrimaryAngles());
			double range = computeScanRange() * Math.PI / 180;
			if (debug) System.out.println("Angular Range: " + range * 180 / Math.PI);
			if (range < maxRange) {
				//delta = (range - Math.PI) / 2;
				deltaX = (range - Math.PI) - delta;
				if (deltaX < 0){
					if (debug) System.out.println("Scan Range was less than 180 + half fan!!!" + deltaX / Math.PI * 180 + " " + (delta / Math.PI * 180)  + " " + range);
				}
			} else {
				// In this case delta is negative:
				deltaX = maxRange - range;
			}
			offset = 0;
			beta = (getPrimaryAngles()[projNum] - minmax[0]+0.0) * Math.PI / 180;
			beta += offset;
			if (debug) System.out.println("delta: " + delta * 180 / Math.PI);
			if (debug) System.out.println("Angular Offset: " + offset * 180 / Math.PI);
			if (debug) System.out.println("Beta: " + beta * 180 / Math.PI);
		} else {
			beta = (5.0 /180 * Math.PI ) +(((projNum + 0.0) / numberOfProjections) + 0.001) / (Math.PI / 180 * 198.0);
			if (debug) System.out.println("Beta: " + beta * 180 / Math.PI);
		}
		return computeParkerWeights1D(beta);
	}

	private double sFunction(double betaprime){
		double revan = 0;
		if (Math.abs(Math.PI) < 0.5) revan = (0.5 *(1 + Math.sin(Math.PI * betaprime)));
		if (betaprime >= 0.5) revan = 1;
		return revan;
	}
	
	private double weight_Wesarg(double alpha, double beta){
		double revan = 0;
		double alphaB = (2*delta) - (2*alpha) - deltaX;
		double nalphaB = (2*delta) - (-2*alpha) - deltaX;
		double alphab = percent * alphaB;
		double nalphab = percent * nalphaB;
		
		revan = 0.5 * ( sFunction((beta / alphab) - 0.5) 
				+ sFunction(((beta - alphaB)/ alphab) + 0.5)
				- sFunction(((beta-Math.PI + (2*alpha))/nalphab)-0.5)
				- sFunction(((beta-Math.PI-(2*delta) + deltaX)/nalphab)+0.5)
		);
		return revan;
	}
	
	protected double computeWesargWeight(double alpha, double beta){
		if (beta < 0){
			beta += 2*Math.PI;
		}
		checkDelta();
		double weight = 1;
		filter = true;
		weight = weight_Wesarg(alpha, beta);
		return weight;
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		Grid2D theFilteredProcessor = imageProcessor;
		filter = false;
		double [] theWeights = this.computeParkerWeights1D(this.imageIndex);
		System.out.println(numberOfProjections);
		if (filter) {
			theWeights = DoubleArrayUtil.gaussianFilter(theWeights, 20);
		}
		if ((imageIndex < 5) || (imageIndex > this.numberOfProjections -5)) {
			//if (debug) VisualizationUtil.createPlot("Projection " + imageIndex, theWeights).show();
		}
		if (debug) DoubleArrayUtil.saveForVisualization(imageIndex, theWeights);
		for (int j = 0; j < theFilteredProcessor.getHeight(); j++){
			for (int i=0; i < theFilteredProcessor.getWidth() ; i++){
				if (theFilteredProcessor.getWidth() <= theWeights.length) {
					double value = theFilteredProcessor.getPixelValue(i, j) * theWeights[i];
					theFilteredProcessor.putPixelValue(i, j, value);
				} else {
					int offset = (theFilteredProcessor.getWidth() - theWeights.length) / 2;
					if (((i-offset) > 0)&&(i-offset < theWeights.length)){
						double value = theFilteredProcessor.getPixelValue(i, j) * theWeights[i - offset];
						theFilteredProcessor.putPixelValue(i, j, value);
					}
				}
			}
		}
		return theFilteredProcessor;
	}

	@Override
	public void configure() throws Exception {
		super.configure();
		percent = UserUtil.queryDouble("Enter q value (q == 1 => ParkerWeights):", percent);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Parker82-OSS,\n" +
		"  author = {{Wesarg}, S. and {Ebert}, M. and {Bortfeld}, T.},\n" +
		"  title = \"{{Parker weights revisited}}\",\n" +
		"  journal = {Medical Physics},\n" +
		"  year = 2002,\n" +
		"  volume = 29,\n"+
		"  number = 3,\n" +
		"  pages = {372-378}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		String medline = "Wesarg S, Ebert M, Bortfeld T. Parker weights revisited. Med. Phys. Volume 29, Issue 3, pp. 372-378 (March 2002) ";
		return medline;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
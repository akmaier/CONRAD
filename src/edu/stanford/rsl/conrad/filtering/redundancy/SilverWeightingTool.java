package edu.stanford.rsl.conrad.filtering.redundancy;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;


public class SilverWeightingTool extends ParkerWeightingTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2453380390532119284L;
	/**
	 * 
	 */
	private double deltaX = 0;
	
	@Override
	public IndividualImageFilteringTool clone() {
		SilverWeightingTool clone = new SilverWeightingTool();
		clone.setDetectorWidth(this.getDetectorWidth());
		clone.setPixelDimensionX(this.getPixelDimensionX());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setNumberOfProjections(numberOfProjections);
		clone.setPrimaryAngles(getPrimaryAngles());
		clone.configured = configured;
		clone.offset = offset;
		clone.deltaX = deltaX;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Silver Redundancy Weighting Filter";
	}

	
	

	@Override
	protected boolean checkBeginCondition(double alpha, double beta){
		double reference = (deltaX) + (2*alpha) ;
		return (0 <= beta +CONRAD.SMALL_VALUE) && (beta  <= reference);
	}
	
	@Override
	protected boolean checkEndCondition(double alpha, double beta){
		return ((Math.PI + (2*alpha)) <= beta +CONRAD.SMALL_VALUE) && (beta <= (Math.PI + deltaX + offset));
	}
	
	@Override
	protected double computeParkerWeight(int x, double beta){
		double value = (this.pixelDimensionX * (x - (detectorWidth/2)));
		double alpha = Math.atan(value / sourceToDetectorDistance);
		return computeOscarWeight(alpha, beta);
	}

	
	protected double computeOscarWeight(double alpha, double beta){
		if (beta < 0){
			beta += 2.0*Math.PI;
		}
		checkDelta();
		if (beta == offset) beta += CONRAD.SMALL_VALUE;
		double weight = 1;
		//if (beta == 0) beta = Function.SMALL_VALUE;
		if (beta-offset < (delta + delta + deltaX)){
			if (checkBeginCondition(alpha, beta)){
				// begin of sweep
				weight =beginWeight(alpha, beta);
				double betaprime = beta + Math.PI - (2*alpha);
				//weight += endWeight(-alpha, betaprime);
				if (!checkEndCondition(-alpha, betaprime)){
					//weight = 2;
				}

			} 
		}
		if (beta > (Math.PI - 2 * delta)){
			if(checkEndCondition(alpha, beta)){
				// end of sweep
				weight = endWeight(alpha, beta);
				double betaprime = beta + Math.PI - (2*alpha);
				if (betaprime > 2.0 * Math.PI) betaprime -= Math.PI *2.0;
				//weight += beginWeight(-alpha,betaprime);
				
				if (!checkBeginCondition(-alpha, betaprime)){
					//weight = 2;
				}
			} 
		}
		//if (weight < 0) weight = 0;
		return weight;
	}

	double numerical = 0.1;
	
	@Override
	protected double beginWeight(double alpha, double beta){
		return Math.pow(Math.sin((Math.PI/2.0) * ((beta) / (deltaX+(2.0*alpha)))),2);
	}
	
	@Override
	protected double endWeight(double alpha, double beta){
		return Math.pow(Math.sin((Math.PI/2.0) * ((Math.PI + (deltaX) - beta)/(deltaX-(2.0*alpha)))), 2);
	}
	


	
	


	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		Grid2D theFilteredProcessor = imageProcessor;
		double [] theWeights = this.computeParkerWeights1D(this.imageIndex);
		//System.out.println(numberOfProjections);
		if (imageIndex < this.numberOfProjections/2) {
			//forceMonotony(theWeights, true);
		} else {
			//forceMonotony(theWeights, false);
		}
		if (imageIndex == 0) {
			//theWeights = DoubleArrayUtil.gaussianFilter(theWeights, 15);
		}
		if (debug) DoubleArrayUtil.saveForVisualization(imageIndex, theWeights);
		if ((imageIndex < 5) || (imageIndex > this.numberOfProjections -5)) {
			//if (debug) VisualizationUtil.createPlot("Projection " + imageIndex, theWeights).show();
		}
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
		Configuration config = Configuration.getGlobalConfiguration(); 
		setConfiguration(config);
//		double [] minmax = null;
		checkDelta();
		setNumberOfProjections(config.getGeometry().getPrimaryAngles().length);
		double maxRange = (Math.PI + (2 * delta));
		if (getPrimaryAngles() != null){
			//minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
			double factor = 15 - (deltaX /Math.PI *180);
			if (factor < 1) factor = 1;
			double range = (computeScanRange() + (factor* config.getGeometry().getAverageAngularIncrement()) ) * Math.PI / 180.0;
			offset = ((maxRange - range) /2);
			offset = factor* config.getGeometry().getAverageAngularIncrement() / 180.0 * Math.PI /2;
			deltaX = range - Math.PI;
			if (debug) System.out.println("delta: " + delta * 180 / Math.PI);
			if (debug) System.out.println("Angular Offset: " + offset + " " + maxRange + " " + range);
			if (debug) System.out.println("deltaX: " + deltaX);
		}


		if (this.numberOfProjections == 0){
			throw new Exception("Number of projections not known");
		}
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Silver00-AMF,\n" +
		"  author = {{Silver}, M. D.},\n" +
		"  title = \"{{A method for including redundant data in computed tomography}}\",\n" +
		"  journal = {Medical Physics},\n" +
		"  year = 2000,\n" +
		"  volume = 27,\n"+
		"  number = 4,\n" +
		"  pages = {773-774}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		String medline = "Silver MD. A method for including redundant data in computed tomography. Med Phys. 2000 27(4):773-4.";
		return medline;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

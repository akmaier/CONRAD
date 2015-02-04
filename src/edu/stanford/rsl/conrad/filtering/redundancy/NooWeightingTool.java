package edu.stanford.rsl.conrad.filtering.redundancy;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;


public class NooWeightingTool extends ParkerWeightingTool {
	/**
	 * 
	 */
	private static final long serialVersionUID = -9073647229814074372L;

	@Override
	public IndividualImageFilteringTool clone() {
		NooWeightingTool clone = new NooWeightingTool();
		clone.setDetectorWidth(this.getDetectorWidth());
		clone.setPixelDimensionX(this.getPixelDimensionX());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setNumberOfProjections(numberOfProjections);
		clone.setPrimaryAngles(getPrimaryAngles());
		clone.offset = offset;
		clone.configured = this.configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Noo Redundancy Weighting Filter";
	}

	
	


	@Override
	protected double computeParkerWeight(int x, double beta){
		double value = (this.pixelDimensionX * (x - (detectorWidth/2)));
		double alpha = Math.atan(value / sourceToDetectorDistance);
		return computeOscarWeight(alpha, beta);
	}

	
	protected double computeOscarWeight(double alpha, double beta){
		//if (beta < 0){
		//	beta += 2.0*Math.PI;
		//}
		checkDelta();
		// In Parkers formulation lambda is beta and alpha is corresponding angle to \tilde{u}
		double weight = 1;
		if (checkBeginCondition(alpha, beta+offset)) {
			weight = parkerLikeWeight(beta - offset, -alpha);;
		}
		if (checkEndCondition(alpha, beta - offset)){
			weight = parkerLikeWeight(beta - offset, -alpha);;
		}
		//if (weight < 0) weight = 0;
		return weight;
	}

	double numerical = 0.1;
	private double deltaX;
	
	protected double parkerLikeWeight(double lambda, double utildeAngle){  
		return cFunction(lambda) / (cFunction(lambda) + cFunction(lambda + Math.PI-(2*utildeAngle)));
	}
		
		
	protected double cFunction(double lambda){
		if (lambda > Math.PI * 2) lambda -= Math.PI * 2;
		if (lambda < 0) lambda += Math.PI * 2;
		double d = 10.0*Math.PI/180;
		double revan = 1;
		double [] minmax = DoubleArrayUtil.minAndMaxOfArray(getPrimaryAngles());
		minmax[0] /= 180.0 / Math.PI;
		minmax[1] /= 180.0 / Math.PI;
		minmax[1] -= minmax[0];
		minmax[0] = 0;
		if ((lambda < minmax[0]+d)&&(lambda > minmax[0])) {
			revan = Math.cos(Math.PI*(lambda-minmax[0]+d)/(2*d));
		}
		if ((lambda > minmax[1]-d)&&(lambda < minmax[1])) {
			revan = Math.cos(Math.PI*((lambda-minmax[1])-d)/(2*d));
		}
		revan = Math.pow(revan, 2);
		return revan;
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
		checkDelta();
		//double [] minmax = null;
		checkDelta();
		setNumberOfProjections(config.getGeometry().getPrimaryAngles().length);
		if (getPrimaryAngles() != null){
			//minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
			double factor = 0;
			double range = (computeScanRange() + (factor* config.getGeometry().getAverageAngularIncrement()) ) * Math.PI / 180.0;
			offset = ((Math.PI + 2*delta) - range) / 2.0;
			deltaX = range - Math.PI;
			if (debug) System.out.println("delta: " + delta * 180 / Math.PI);
			if (debug) System.out.println("deltaX: " + deltaX);
		}
		if (this.numberOfProjections == 0){
			throw new Exception("Number of projections not known");
		}
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Noo02-IRF,\n" +
		"  author = {Noo, F. and  Defrise, M. and Clackdoyle, R. and Kudo, H.},\n" +
		"  title = \"{{Image reconstruction from fan-beam projections on less than a short scan}}\",\n" +
		"  journal = {Physics in Medicine and Biology},\n" +
		"  year = 2002,\n" +
		"  volume = 47,\n"+
		"  number = 14,\n" +
		"  pages = {2525-2546}\n" +
		"}";
		return bibtex;
	}
	
	@Override
	public String getMedlineCitation(){
		return "Noo F, Defrise M, Clackdoyle R, Kudo H. Image reconstruction from fan-beam projections on less than a short scan. " +
				"Phys Med Biol 47(14):2525-46. 2002.";
	}
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
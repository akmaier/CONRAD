package edu.stanford.rsl.conrad.filtering.redundancy;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;


public class RiessWeightingTool extends ParkerWeightingTool {

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
		RiessWeightingTool clone = new RiessWeightingTool();
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
		return "Riess Redundancy Weighting Filter";
	}

	
	

	/**
	 * enables a shift from no Parker Weighting to full Parker Weighting. One Can choose: how strong the remaining stripes will be ...
	 * 
	 * @param alpha
	 * @param beta
	 * @param tau
	 * @return the interpolated Parker weight
	 */
	protected double computeParkerWeight_shift(double alpha, double beta, double tau){
		if (beta < 0){
			beta += 2*Math.PI;
		}
		checkDelta();
		double weight = 1 - tau;
		//if (beta == 0) beta = OSCAR.SMALL_VALUE;
		if (beta < ((delta + deltaX) - (2*alpha))){
			// begin of sweep
			double Nweight = beginWeight(alpha, beta); 
			if (beta < (delta + delta + delta + deltaX)){
				double test = Math.exp(- 0.5 * Math.pow((((delta + deltaX) - (2*alpha))  - beta) / 0.02 ,2));
				double difference = (weight - Nweight);
				if (difference > 0 ) {
					weight = Nweight + difference * test;
				}
			}
		} else if((beta >= (Math.PI - (2*alpha) )) && (beta <= (Math.PI + delta + deltaX ))){
			// end of sweep
			double Nweight = endWeight(alpha, beta);
			if (beta > (Math.PI - 2 * delta)){
				double test = Math.exp(- 0.5 * Math.pow(((Math.PI - (2*alpha)) - beta) / 0.02 ,2));
				double difference = (weight - Nweight);
				if (difference > 0 ) {
					weight = Nweight + difference * test;
				}
			}
		}
		weight += tau;
		return weight;
	}



	
	@Override
	protected double beginWeight(double alpha, double beta){
		double revan = polynomialWrapper1(linearFunctionMinusOneZero(2*alpha + deltaX, 0, beta));
		//double center = (alpha - (delta)) / (delta - (beta-deltaX)/2);
		//double center = (beta + (2*delta) - (2*alpha))/(deltaX+(2*delta));
		//revan += Math.pow(Math.sin((Math.PI/2.0) * center), 2);	
		return revan;
	}
	
	protected double OSCARBeginWeight(double alpha, double beta){
		double revan = 1+ polynomialWrapper1(linearFunctionMinusOneZero(0, -deltaX - 2*alpha, beta));
		return revan;
	}
	
	@Override
	protected double endWeight(double alpha, double beta){
		double revan =polynomialWrapper2(linearFunctionMinusOneZero(Math.PI+deltaX, Math.PI + (2*alpha), beta));
		
		//double center = (alpha - delta) / (delta - (beta-Math.PI)/2);
		//double center = (deltaX-beta-Math.PI)/(deltaX+(2*delta));
		//revan += Math.pow(Math.sin((Math.PI/2.0) * center), 2);	
		return revan;
	}
	
	protected double OSCAREndWeight(double alpha, double beta){
		double revan = 1;
		revan += polynomialWrapper2(linearFunctionMinusOneZero(Math.PI + 2*deltaX - (2*alpha), Math.PI+deltaX, beta));
		return revan;
	}
	
	protected double polynomialWrapper1(double x){
		return (0.25*Math.pow(2*x+1,3))-(1.5*x)-0.25;
	}
	
	protected double polynomialWrapper2(double x){
		return -(0.25*Math.pow(2*x+1,3))+(1.5*x)+1.25;
	}
	
	protected double linearFunctionMinusOneZero(double one, double zero, double x){
		double m = -1.0/(one-zero);
		double t = -m*zero;
		return m*x+t;
	}
	
	@Override
	protected boolean checkBeginCondition(double alpha, double beta){
		double reference = (deltaX) + (2*alpha);
		return (0 <= beta ) && (beta  <= reference);
	}
	
	protected boolean checkBeginConditionOuter(double alpha, double beta){
		double reference = (-deltaX) - (2*alpha) ;
		return (0 <= beta) && (beta  <= reference);
	}
	
	@Override
	protected boolean checkEndCondition(double alpha, double beta){
		return ((Math.PI + (2*alpha) ) <= beta ) && (beta <= (Math.PI + deltaX ));
	}
	
	protected boolean checkEndConditionOuter(double alpha, double beta){
		return ((Math.PI + 2*deltaX - (2*alpha)) <= beta) && (beta <= (Math.PI + deltaX ));
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
		//if (beta == 0) beta += OSCAR.SMALL_VALUE;
		double weight = 1;
		//if (beta == 0) beta = Function.SMALL_VALUE;
		if (beta < (delta + delta + delta + delta)){
			if (checkBeginCondition(alpha, beta)){
				// begin of sweep
				weight =beginWeight(alpha, beta);
				double betaprime = beta + Math.PI - (2*alpha);
				//weight += endWeight(-alpha, betaprime);
				if (!checkEndCondition(-alpha, betaprime)){
					//weight = 2;
				}

			}
			if (checkBeginConditionOuter(alpha,beta)){
				weight = this.OSCARBeginWeight(alpha, beta);
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
			if (checkEndConditionOuter(alpha,beta)){
				weight = OSCAREndWeight(alpha, beta);
			}
		}
		if (beta > Math.PI + 2 * this.delta) weight = 0;
		//if (weight < 0) weight = 0;
		return weight;
	}

	double numerical = 0.1;
	

	

	
	/**
	 * Correct End weight according to the annotation in Kak & Slaney
	 * @param alpha
	 * @param beta
	 * @return the end weight
	 */
	protected double endWeight_ParkerLike(double alpha, double beta){
		return Math.pow(Math.sin((Math.PI/4) * ((Math.PI + (delta + delta) - beta)/(delta-alpha))), 2);
	}
	
	/**
	 * Correct Parker Weight according to the annotation in Kak and Slaney
	 * @param alpha
	 * @param beta
	 * @return the begin weight
	 */
	protected double beginWeight_ParkerLike(double alpha, double beta){
		return Math.pow(Math.sin((Math.PI/4) * ((beta) / (delta+alpha))),2);
	}

	protected double beginWeight_Polynom1(double alpha, double beta){
		double value = Math.PI / 2;
		double b = (1- ((alpha-delta)/Math.PI)) * 4;
		value -= ((2*(delta+alpha))-beta) /b;
		return Math.pow(Math.sin(value), 2);
	}
	
	protected double endWeight_Polynom1(double alpha, double beta){
		double value = Math.PI / 2;
		double b = (1- ((alpha-delta)/Math.PI)) * 4;
		double a = -b;
		value -= (Math.PI +(2*(alpha))-beta) /a;
		return Math.pow(Math.sin(value), 2);
	}
	
	
	


	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		Grid2D theFilteredProcessor = imageProcessor;
		
		double [] theWeights = null;
		try {
			theWeights = this.computeParkerWeights1D(this.imageIndex);
		} catch (Exception e){
			System.out.println("No angle for projection " + imageIndex + ". Paint it black.");
			theWeights = new double [imageProcessor.getWidth()];
		}
		//System.out.println(numberOfProjections);
		if (imageIndex < this.numberOfProjections/2) {
			//forceMonotony(theWeights, true);
		} else {
			//forceMonotony(theWeights, false);
		}
		if (true) {
			
		}
		if (debug) DoubleArrayUtil.saveForVisualization(imageIndex, theWeights);
		if ((imageIndex < 5) || (imageIndex > this.numberOfProjections -5)) {
			//if (debug) VisualizationUtil.createPlot("Projection old" + imageIndex, theWeights).show();
			for (int i= 0; i < theWeights.length; i++){
				if (Double.isNaN(theWeights[i])) theWeights[i] = 0;
			}
			theWeights = DoubleArrayUtil.gaussianFilter(theWeights, 20);
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
		double [] minmax = null;
		checkDelta();
		setNumberOfProjections(config.getGeometry().getPrimaryAngles().length);
		double maxRange = (Math.PI + (2 * delta));
		if (getPrimaryAngles() != null){
			minmax = DoubleArrayUtil.minAndMaxOfArray(getPrimaryAngles());
			double factor = 1;
			if (factor < 1) factor = 1;
			double range = (minmax[1] - minmax[0] + (factor* config.getGeometry().getAverageAngularIncrement()) ) * Math.PI / 180.0;
			offset = ((maxRange - range) /2);
			offset = factor* config.getGeometry().getAverageAngularIncrement() / 180.0 * Math.PI /2;
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
		String bibtex = new String("@inproceedings{Riess2013-TNT," +
			"author = {Riess, Christian and Berger, Martin and Wu, Haibo and Manhart, Michael and Fahrig, Rebecca and Maier, Andreas},"+
			"booktitle = {Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine},"+
			"editor = {Leahy, Richard M and Qi, Jinyi},"+
			"pages = {341--344},"+
			"title = {{TV or not TV? That is the Question}},"+
			"year = {2013}");
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return new String("Riess, C., Berger, M., Wu, H., Manhart, M., Fahrig, R., & Maier, A. (2013). " +
				"TV or not TV? That is the Question. In R. M. Leahy & J. Qi (Eds.), " +
				"Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine (pp. 341�344).");
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
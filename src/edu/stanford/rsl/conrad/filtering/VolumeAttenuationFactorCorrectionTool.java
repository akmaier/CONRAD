package edu.stanford.rsl.conrad.filtering;


import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.physics.Constants;
import edu.stanford.rsl.conrad.utils.UserUtil;



/**
 * This Class implements the first step of the beam hardening correction after Joseph & Spittal.
 * In order to perform the correction the filter needs to be configured. To do so a measurement
 * of a water corrected reconstruction is required.<br>
 * Both soft tissue, i.e. water and the hard material, e.g. bone or some kind of metal, have to
 * be measured from a preliminary reconstruction. These values plus a threshold which removes
 * any soft tissue and artifact from the image are supplied to this filtering step<br> 
 * The result of this filtering step should contain only the hard tissue with its correct density
 * value.
 * 
 * @author akmaier
 * @see edu.stanford.rsl.conrad.cuda.CUDAForwardProjector
 * @see ApplyLambdaWeightingTool
 *
 */
public class VolumeAttenuationFactorCorrectionTool extends IndividualImageFilteringTool {


	/**
	 * 
	 */
	private static final long serialVersionUID = -4178589044274723460L;
	/**
	 * 
	 */
	private double minimum = 2.0;
	private double measureHard = 2.7;
	private double measureSoft = 1.0;
	boolean debug = true;
	
	
	@Override
	public IndividualImageFilteringTool clone() {
		VolumeAttenuationFactorCorrectionTool clone = new VolumeAttenuationFactorCorrectionTool();
		clone.minimum = minimum;
		clone.measureHard = measureHard;
		clone.measureSoft = measureSoft;
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Volume Attenuation Factor Correction";
	}

	/**
	 * Computes the lambda_0 factor according to Joseph and Spital.
	 * @param measureHard as density
	 * @param measureSoft as density
	 * @return lambda_0 the lambda coefficient
	 */
	public static double getLambda0(double measureHard, double measureSoft){
		double hard = measureHard;//Constants.computeMassDensity(measureHard);
		double soft = measureSoft;//Constants.computeMassDensity(measureSoft);
		return (hard * Constants.SoftTissue_mdensity) / (soft * Constants.Al_mdensity);
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		Grid2D imp = new Grid2D(imageProcessor.getWidth(), imageProcessor.getHeight());
		double lambda0 = getLambda0(measureHard, measureSoft);
		double minScale = minimum / lambda0;
		if (debug) System.out.println("Estimated correction factor: " + lambda0);
		for (int i = 0; i < imageProcessor.getWidth(); i++){
			for (int j = 0; j < imageProcessor.getHeight(); j++){
				double value = imageProcessor.getPixelValue(i,j);
				value /= lambda0;
				if (value < minScale) {
					value = 0;
				}
				imp.putPixelValue(i, j, value);
			}
		}
		return imp;
	}


	@Override
	public void configure(){
		try {
			measureHard = UserUtil.queryDouble("Measurement of hard material [g/cm^3]", measureHard);
			measureSoft= UserUtil.queryDouble("Measurement of soft material [g/cm^3]", measureSoft);
			minimum = UserUtil.queryDouble("Cut-off for hard material [g/cm^3]", minimum);
			configured = true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Zellerhof05-LC3,\n" +
		"  author = {{Joseph}, P. M. and {Spital}, R. D.},\n" +
		"  title = {{A method for correcting bone induced artifacts in computed tomography scanners}},\n" +
		"  booktitle = {{Journal of Computer Assisted Tomography}},\n" +
		"  volume = {2}, " +
		"  pages= {100-108},\n" +
		"  year = {1978}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Joseph PM, Spital RD. A method for correcting bone induced artifacts in computed tomography scanners, JCAT 1978;2:100-8.";
	}



	/**
	 * beam hardening is also device dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return true;
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.StatisticsUtil;


/**
 * Applies Poisson noise to the input image. The pixel-wise lambda is assumed to be the value of each input pixel.
 * 
 * @author Andreas Maier
 * @see edu.stanford.rsl.conrad.filtering.LogPoissonNoiseFilteringTool
 *
 */
public class PoissonNoiseFilteringTool extends IndividualImageFilteringTool {

	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 647343713286585178L;

	@Override
	public IndividualImageFilteringTool clone() {
		PoissonNoiseFilteringTool filter = new PoissonNoiseFilteringTool();
		filter.configured = configured;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Poisson Noise";
	}

	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		Grid2D imp = new Grid2D(imageProcessor.getWidth(), imageProcessor.getHeight());
		for (int k = 0; k < imageProcessor.getWidth(); k++){
			for (int j = 0; j < imageProcessor.getHeight(); j++){
				double value = StatisticsUtil.poissonRandomNumber(imageProcessor.getPixelValue(k, j));
				imp.putPixelValue(k, j, value);
			}
		}
		return imp;
	}
	
	

	public void prepareForSerialization(){
		super.prepareForSerialization();
	}
	
	
	@Override
	public void configure() throws Exception {
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@article{Atkinson79-TCG,\n  author={Atkinson AC},\n  title={The Computer Generation of Poisson Random Variables.},\n  journal={Journal of the Royal Statistical Society. Series C (Applied Statistics)},\n" +
				"  volume={28},\n  number={1},  pages={29-35},\n  year={1979}\n}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "A.C. Atkinson. The Computer Generation of Poisson Random Variables. " + 
	 "Journal of the Royal Statistical Society. Series C (Applied Statistics) " +
	 "Vol. 28, No. 1 (1979), pp. 29-35";
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
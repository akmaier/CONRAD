package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.fitting.Function;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;


/**
 * This Class can be used to apply a previously learned Hounsfield scaling to an existing
 * reconstruction. 
 * 
 * @author Chris Schwemmer
 * @see edu.stanford.rsl.conrad.reconstruction.ReconstructionFilter
 *
 */
public class ApplyHounsfieldScaling extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6671943273538431531L;

	@Override
	public void configure() throws Exception {
		configured = true;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		ApplyHounsfieldScaling clone = new ApplyHounsfieldScaling();
		
		clone.configured = configured;
		
		return clone;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imp)
			throws Exception {
		Function hounsfield = Configuration.getGlobalConfiguration().getHounsfieldScaling();
		
		for (int i = 0; i < imageProcessor.getWidth(); i++){
			for (int j = 0; j < imageProcessor.getHeight(); j++){
				double value = hounsfield.evaluate(imageProcessor.getPixelValue(i, j));
				
				if (value < -1024)
					value = -1024;
				
				imp.putPixelValue(i, j, value);
			}
		}
		
		return imp;
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getToolName() {
		return "Apply Hounsfield Scaling";
	}

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return CONRAD.CONRADMedline;
	}
}
/*
 * Copyright (C) 2010-2014 - Chris Schwemmer 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
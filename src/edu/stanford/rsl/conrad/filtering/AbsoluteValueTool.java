/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class AbsoluteValueTool extends IndividualImageFilteringTool {


	


	/**
	 * 
	 */
	private static final long serialVersionUID = 260472749566522423L;

	public AbsoluteValueTool (){
		configured = true;
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		NumericPointwiseOperators.abs(imageProcessor);
		return imageProcessor;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new AbsoluteValueTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Absolute Value Tool";
	}

	@Override
	public void configure() throws Exception {
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	/**
	 * Is not device, but pipeline dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}

}



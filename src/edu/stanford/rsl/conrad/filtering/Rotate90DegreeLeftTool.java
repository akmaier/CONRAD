/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import ij.process.FloatProcessor;

public class Rotate90DegreeLeftTool extends IndividualImageFilteringTool {


	
	/**
	 * 
	 */
	private static final long serialVersionUID = 865405590293794136L;

	public Rotate90DegreeLeftTool (){
		configured = true;
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		FloatProcessor in = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		in.setPixels(imageProcessor.getBuffer());
		FloatProcessor left = (FloatProcessor) in.rotateLeft();
		Grid2D out = new Grid2D((float[])left.getPixels(), left.getWidth(), left.getHeight());
		return out;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new Rotate90DegreeLeftTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Rotate 90 Degree Left Tool";
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

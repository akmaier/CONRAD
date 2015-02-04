package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class HorizontalFlippingTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -727180365181022869L;

	
	public HorizontalFlippingTool (){
		configured = true;
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D image) {
		ImageProcessor imageProcessor = new FloatProcessor(image.getWidth(), image.getHeight());
		imageProcessor.setPixels(image.getBuffer());
		imageProcessor.flipHorizontal();
		return image;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new HorizontalFlippingTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Horizontal Flipping Tool";
	}

	@Override
	public void configure() throws Exception {
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@BOOK{OMalley52-LOT,\n" +
				"  author = {{O'Malley}, C. D. and {Sounders}, C. M.},\n" +
				"  title = {{Leonardo on the Human Body: The Anatomical, Physiological, and Embryological Drawings of Leonardo da Vinci. With Translations, Emendations and a Biographical Introduction}},\n" +
				"  publisher = {Henry Schuman},\n" +
				"  address = {New York, United States},\n" +
				"  year = {1952}\n" +
				"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "O'Malley CD, Sounders CM. Leonardo on the Human Body: The Anatomical, Physiological, and Embryological Drawings of Leonardo da Vinci. With Translations, Emendations and a Biographical Introduction. Henry Schuman, New York, United States 1952.";
	}

	/**
	 * Is not device, but pipeline dependent.
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

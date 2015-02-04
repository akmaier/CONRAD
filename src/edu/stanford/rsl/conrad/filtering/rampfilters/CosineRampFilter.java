package edu.stanford.rsl.conrad.filtering.rampfilters;

public class CosineRampFilter extends RampFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8510178767243770599L;

	@Override
	public RampFilter clone() {
		RampFilter clone = new CosineRampFilter();
		clone.setSourceToAxisDistance(this.getSourceToAxisDistance());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setCutOffFrequency(this.getCutOffFrequency());
		clone.setPhysicalPixelWidthInMilimeters(this.getPhysicalPixelWidthInMilimeters());
		return clone;
	}

	@Override
	public double getFilterWeight(double ku) {
		double revan = 1;
		if (ku != 0) revan = Math.cos(ku / (2 * cutOffFrequency));
		return revan;
	}

	@Override
	public String getRampName() {
		return "Cosine Ramp Filter";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
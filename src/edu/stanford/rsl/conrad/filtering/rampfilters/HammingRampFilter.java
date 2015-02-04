package edu.stanford.rsl.conrad.filtering.rampfilters;

public class HammingRampFilter extends RampFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8909229941245077452L;

	@Override
	public RampFilter clone() {
		RampFilter clone = new HammingRampFilter();
		clone.setCutOffFrequency(this.getCutOffFrequency());
		clone.setSourceToAxisDistance(this.getSourceToAxisDistance());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setPhysicalPixelWidthInMilimeters(this.getPhysicalPixelWidthInMilimeters());
		return clone;
	}

	@Override
	public double getFilterWeight(double ku) {
		double revan = 1;
		if (ku != 0) revan = 0.53836 + 0.46164 * Math.cos(ku / (cutOffFrequency));
		return revan;
	}

	@Override
	public String getRampName() {
		return "Hamming Ramp Filter";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.rampfilters;

public class HanningRampFilter extends RampFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4538976511084471631L;

	@Override
	public RampFilter clone() {
		RampFilter clone = new HanningRampFilter();
		clone.setSourceToAxisDistance(this.getSourceToAxisDistance());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setCutOffFrequency(this.getCutOffFrequency());
		clone.setPhysicalPixelWidthInMilimeters(this.getPhysicalPixelWidthInMilimeters());
		return clone;
	}

	@Override
	public double getFilterWeight(double ku) {
		double revan = 1;
		if (ku != 0) revan = (1 + Math.cos(ku / (cutOffFrequency))) / 2;
		return revan;
	}

	@Override
	public String getRampName() {
		return "Hanning Ramp Filter";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
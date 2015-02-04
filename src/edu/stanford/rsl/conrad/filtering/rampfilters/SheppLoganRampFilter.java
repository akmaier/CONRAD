package edu.stanford.rsl.conrad.filtering.rampfilters;

public class SheppLoganRampFilter extends RampFilter {


	/**
	 * 
	 */
	private static final long serialVersionUID = 8055940389460979441L;

	@Override
	public RampFilter clone() {
		SheppLoganRampFilter clone = new SheppLoganRampFilter();
		clone.setSourceToAxisDistance(this.getSourceToAxisDistance());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setCutOffFrequency(this.getCutOffFrequency());
		clone.setPhysicalPixelWidthInMilimeters(this.getPhysicalPixelWidthInMilimeters());
		return clone;
	}

	@Override
	public double getFilterWeight(double ku) {
		double revan = 1;
		if (ku != 0) revan = Math.sin(ku / (2 * cutOffFrequency)) / (ku / (2 * cutOffFrequency));
		return revan;
	}

	@Override
	public String getRampName() {
		return "Shepp-Logan Ramp Filter";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

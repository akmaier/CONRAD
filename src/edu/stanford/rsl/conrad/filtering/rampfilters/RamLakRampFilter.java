package edu.stanford.rsl.conrad.filtering.rampfilters;

/**
 * Class to represent a Ram-Lak RampFilter. It contains only entries of 1. Hence, it is ideal for testing;
 * 
 * @author Andreas Maier
 *
 */
public class RamLakRampFilter extends RampFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -372550797243795367L;

	@Override
	public RampFilter clone() {
		RampFilter clone = new RamLakRampFilter();
		clone.setSourceToAxisDistance(this.getSourceToAxisDistance());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setCutOffFrequency(this.getCutOffFrequency());
		clone.setPhysicalPixelWidthInMilimeters(this.getPhysicalPixelWidthInMilimeters());
		return clone;
	}

	@Override
	public String getRampName() {
		return "Ram-Lak Ramp Filter";
	}

	@Override
	public double getFilterWeight(double ku) {
		return 1;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

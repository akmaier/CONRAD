package edu.stanford.rsl.conrad.filtering.rampfilters;

import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class SheppLoganRampFilterWithRollOff extends SheppLoganRampFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4506146824678039341L;
	private double start = 0.628;
	private double scale = 1.8;
	private double slope = 1.0;
	
	
    @Override
    public RampFilter clone() {
          SheppLoganRampFilterWithRollOff clone = new SheppLoganRampFilterWithRollOff();
          clone.setSourceToAxisDistance(this.getSourceToAxisDistance());
          clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
          clone.setCutOffFrequency(this.getCutOffFrequency());
          clone.setPhysicalPixelWidthInMilimeters(this.getPhysicalPixelWidthInMilimeters());
          clone.start=start;
          clone.scale=scale;
          clone.slope=slope;
          return clone;
    }

    @Override
    public String getRampName() {
          return "Shepp-Logan Ramp Filter with roll-off (Start "+start+", Scale "+scale +", Slope "+slope+")";
    }

	public double getStart() {
		return start;
	}

	public void setStart(double start) {
		this.start = start;
	}

	public double getScale() {
		return scale;
	}

	public void setScale(double scale) {
		this.scale = scale;
	}
	
	private double cutoff(double value){
		return Math.pow((1 / (1 + value - start)), scale * (slope *(value -start)));
	}

	@Override
	public double [] getRampFilter1D(int width){
		IRRFilter butter = new IRRFilter();
		double [] filter = super.getRampFilter1D(width);
		butter.setFilterType("LP");
		butter.setPrototype(IRRFilter.BUTTERWORTH);
		butter.setOrder((int) scale);
		butter.setRate(filter.length);
		butter.setFreq1(0.0f);
		butter.setFreq2((float) ((1-start) * (filter.length/2)));
		butter.setFreqPoints((filter.length/4)+1);
		butter.design();
		//System.out.println(butter.filterGain().length);
		float [] butterP = butter.filterGain();
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < butterP.length; i++){
			if (butterP[i] < min) min = butterP[i];
			if (butterP[i] > max) max = butterP[i];
		}
		for (int i = 0; i < (width/2)+1; i++){
			double ku = 2 * Math.PI * i / width;
			// Compute Weight
			if (ku > start) {
				ku = cutoff(ku);
				
			} else {
				ku = 1;
			}
			//ku = ((((butterP[i] - min) / (min - max))) + 0.4);
			// Apply to complex value
			filter[2 * i]  *= ku;
			filter[(2 * i)+1] *= ku;

		}
		DoubleArrayUtil.forceSymmetryComplexDoubleArray(filter);
		return filter;
	}
	
	@Override
	public void configure() throws Exception{
		super.configure();
		start = UserUtil.queryDouble("Enter start value (0-Pi):", start);
		scale = UserUtil.queryDouble("Enter cutoff strength (1.0=linear, 2.0=quadratic)", scale);
		slope = UserUtil.queryDouble("Enter slope of cutoff (0.0 = Shepp-Logan, 0.3 = Normal, 1.0 = Smooth):", slope);
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

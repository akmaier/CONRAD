package edu.stanford.rsl.conrad.filtering.rampfilters;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import edu.stanford.rsl.conrad.utils.FileUtil;


public class ArbitraryRampFilter extends RampFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7312562756045270927L;
	private double[] filter;
	private String file = null;

	public double[] getFilter() {
		return filter;
	}

	public void setFilter(double [] filter) {
		this.filter = filter;
	}

	@Override
	public RampFilter clone() {
		ArbitraryRampFilter filter = new ArbitraryRampFilter();
		filter.setFilter(this.filter);
		return filter;
	}

	@Override
	public double getFilterWeight(double ku) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double [] getRampFilter1D(int width){
		// TODO scale filter in an appropriate manner
		width = filter.length;
		double [] filter = new double[width*2];
		// Apply the filter until the center point
		for (int i = 0; i < width; i++){
			filter[2 * i] = this.filter[i];
		}
		return filter;
	}


	@Override
	public String getRampName() {
		if (file == null) {
			return "Arbitrary Ramp Filter (requires file)";
		} else {
			return "Arbitrary Ramp Filter (" + file +")";
		}
	}

	@Override
	public void configure() throws Exception{
		file = FileUtil.myFileChoose(".txt", false);
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
		filter = (double []) ois.readObject();
		ois.close();
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class ConstantTimeWarper extends TimeWarper {

	/**
	 * Always returns the same time.
	 * @author Chris Schwemmer
	 */
	private static final long serialVersionUID = -8589588541432169559L;
	
	double c;

	public ConstantTimeWarper(double c) {
		super();
		this.c = c;
	}

	@Override
	public double warpTime(double time) {
		return c;
	}

	public static void main (String [] args){
		TimeWarper warp = new ConstantTimeWarper(0.5);
		double [] values = new double [100];
		for (int i =0; i< 100; i++){
			values[i] = warp.warpTime(((double)i)/100);
		}
		VisualizationUtil.createPlot(values).show();
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
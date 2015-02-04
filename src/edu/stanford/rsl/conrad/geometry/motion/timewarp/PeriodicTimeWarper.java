package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import ij.ImageJ;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;



/**
 * Implements a regular periodic time warping. Internally the time is mapped to 0.5 - 0.5* cos(time**2Math.PI). At time = 0.5 the maximal motion is achieved. At time = 1 the motion is returned to the original position.<BR><BR>
 *  <img src = "http://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Sine_cosine_one_period.svg/600px-Sine_cosine_one_period.svg.png" alt="Sine Function">
 * 
 * @author akmaier
 *
 */
public class PeriodicTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7128385122818413431L;

	@Override
	public double warpTime(double time) {
		return 0.5 - 0.5 * Math.cos(time*2*Math.PI);
	}

	public static void main (String [] args){
		TimeWarper warp = new PeriodicTimeWarper();
		double [] values = new double [100];
		for (int i =0; i< 100; i++){
			values[i] = warp.warpTime(((double)i)/100);
		}
		VisualizationUtil.createPlot(values).show();
		new ImageJ();
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
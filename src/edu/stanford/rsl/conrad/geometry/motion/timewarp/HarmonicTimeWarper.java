package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;



/**
 * Implements a harmonic motion which is n x the original Motion.
 * 
 * @author akmaier
 *
 */
public class HarmonicTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7097330258596613543L;
	TimeWarper chirpWarper = new ConstantTimeWarper(1);
	
	/**
	 * Creates a new harmonic time warper. The new motion is n-times the old motion.
	 * @param n
	 */
	public HarmonicTimeWarper(double n){
		this.chirpWarper = new ConstantTimeWarper(n);
	}
	
	public HarmonicTimeWarper(TimeWarper chirpWarper){
		this.chirpWarper = chirpWarper;
	}
	
	@Override
	public double warpTime(double time) {
		double newTime = chirpWarper.warpTime(time)*time;
		if (newTime > 1.0) {
			newTime -= Math.floor(newTime);
		}
		return newTime;
	}

	public static void main (String [] args){
		TimeWarper warp = new HarmonicTimeWarper(2);
		double [] values = new double [100];
		for (int i =0; i< 100; i++){
			values[i] = warp.warpTime(((double)i)/100);
		}
		VisualizationUtil.createPlot(values).show();
		
		warp = new HarmonicTimeWarper(new ScaledIdentitiyTimeWarper(4,16));
		double [] values2 = new double [1000];
		for (int i =0; i< 1000; i++){
			values2[i] = warp.warpTime(((double)i)/1000);
		}
		VisualizationUtil.createPlot(values2).show();
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
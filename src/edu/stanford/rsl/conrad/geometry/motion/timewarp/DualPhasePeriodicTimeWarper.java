package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;



/**
 * 
 * @author akmaier
 *
 */
public class DualPhasePeriodicTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2589628945345492493L;
	protected double phase1;
	
	public DualPhasePeriodicTimeWarper(double portionPhase1, double portionPhase2){
		phase1 = (portionPhase1)/(portionPhase1+portionPhase2);
	}
	
	@Override
	public double warpTime(double time) {
		if (time > phase1){
			double val = Math.PI / (1-phase1);
			val *= (time + 1 - (2*phase1));
			return 0.5 - (0.5 * Math.cos(val));		
		}
		return 0.5 - (0.5 * Math.cos(time*(Math.PI/(phase1))));
	}
	
	public static void main (String [] args){
		TimeWarper warp = new DualPhasePeriodicTimeWarper(2, 3);
		TimeWarper warp2 = new HarmonicTimeWarper(1);
		double [] values = new double [100];
		for (int i =0; i< 100; i++){
			values[i] = warp.warpTime(warp2.warpTime(((double)i)/100));
		}
		VisualizationUtil.createPlot(values).show();
	}

	/**
	 * @return the phase1
	 */
	public double getPhase1() {
		return phase1;
	}

	/**
	 * @param phase1 the phase1 to set
	 */
	public void setPhase1(double phase1) {
		this.phase1 = phase1;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
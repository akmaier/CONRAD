package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;



/**
 * Implements a movement with a rest phase. For the x% around t = 0.5, no movement will occur. The remainder is linearly scaled.
 * The argument of setRestPhase() is a fractional value between 0 and 1.
 * 
 * @author Chris Schwemmer
 *
 */
public class RestPhaseTimeWarper extends TimeWarper {

	private static final long serialVersionUID = -8951035976077197467L;
	private double restPhase = 0.2;
	private double scale = 1.0 / (1.0 - 2.0 * restPhase);

	@Override
	public double warpTime(double time) {
		if (Math.abs(time - 0.5) <= restPhase)
			return 0.5;
		else if (time < 0.5 - restPhase)
			return time * scale;
		else
			return time * scale - (scale - 1.0);
	}
	
	public void setRestPhase(double r) {
		restPhase = r;
		scale = 1.0 / (1.0 - 2.0 * restPhase);
	}
	
	public double getRestPhase() {
		return restPhase;
	}

	public static void main (String [] args) throws Exception{
		RestPhaseTimeWarper warp = new RestPhaseTimeWarper();
		double r = UserUtil.queryDouble("Rest Phase in %", 20);
		warp.setRestPhase(r / 100);
		double [] values = new double [100];
		for (int i =0; i< 100; i++){
			values[i] = warp.warpTime(((double)i)/100);
		}
		VisualizationUtil.createPlot(values).show();
	}
	
}
/*
 * Copyright (C) 2010-2014 Chris Schwemmer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
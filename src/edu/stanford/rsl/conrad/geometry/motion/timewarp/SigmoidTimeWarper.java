package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;



/**
 * Implements an accelerated/deaccelerated time warping. At time = 0.5 the maximal acceleration is achieved.
 * 
 * @author Chris Schwemmer
 *
 */
public class SigmoidTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5822332120091846823L;
	private double acc = 4.0;

	@Override
	public double warpTime(double time) {
		//if (time == 0.0)
		//	return 0.0;
		//else if (time == 1.0)
		//	return 1.0;
		//else
			return  1.0 / (1 + Math.exp(-2.0 * acc * time + acc));
	}
	
	public void setAcc(double accFac) {
		acc = accFac;
	}
	
	public double getAcc() {
		return acc;
	}

	public static void main (String [] args) throws Exception{
		SigmoidTimeWarper warp = new SigmoidTimeWarper();
		double accFac = UserUtil.queryDouble("Acceleration factor", 4);
		warp.setAcc(accFac);
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
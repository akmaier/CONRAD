/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.motion.timewarp;

import java.util.ArrayList;
import java.util.Iterator;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Implements a combined time warper, which sequentielly applies multiple time warpers.
 * This is useful, e.g. for simulating repeated(harmonic), periodic motion.
 * @author Martin Berger
 *
 */
public class CombinedTimeWarper extends TimeWarper {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1651501202588667135L;
	private ArrayList<TimeWarper> warperList;
	
	public CombinedTimeWarper() {
		warperList = new ArrayList<TimeWarper>();
		warperList.add(new IdentityTimeWarper());
	}
	
	public CombinedTimeWarper(TimeWarper... warpers ) {
		warperList = new ArrayList<TimeWarper>();
		for (int i = 0; i < warpers.length; i++) {
			warperList.add(warpers[i]);
		}
	}

	public ArrayList<TimeWarper> getWarperList(){
		return warperList;
	}
	
	public void setWarperList(ArrayList<TimeWarper> wl){
		warperList = wl;
	}

	@Override
	public double warpTime(double time) {
		double t = time;
		if (warperList==null || warperList.size() < 1)
			t=time;
		else{
			Iterator<TimeWarper> wIt = warperList.iterator();
			while (wIt.hasNext()) {
				t = wIt.next().warpTime(t);
			}
		}
		return t;
	}

	public static void main (String [] args) throws Exception{
		SigmoidTimeWarper sigmoid = new SigmoidTimeWarper();
		sigmoid.setAcc(8);
		CombinedTimeWarper warp = new CombinedTimeWarper(new CombinedTimeWarper(new HarmonicTimeWarper(new ScaledIdentitiyTimeWarper(4,32)),new PeriodicTimeWarper(),sigmoid));
		//new CombinedTimeWarper(new HarmonicTimeWarper(new ScaledIdentitiyTimeWarper(4,32)),new PeriodicTimeWarper(),sigmoid);
		double [] values = new double [1000];
		for (int i =0; i< values.length; i++){
			values[i] = warp.warpTime(((double)i)/values.length);
		}
		VisualizationUtil.createPlot(values).show();
	}
	
}

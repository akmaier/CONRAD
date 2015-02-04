package edu.stanford.rsl.conrad.geometry.trajectories;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.utils.Configuration;


public class MultiSweepTrajectory extends
Trajectory {

	/**
	 * 
	 */
	private static final long serialVersionUID = 800597336777372574L;
	
	public MultiSweepTrajectory(Trajectory geometry) {
		super(geometry);
	}


	public void extrapolateProjectionGeometry(){
		int numSweeps = Configuration.getGlobalConfiguration().getNumSweeps();
		if (numSweeps > 1) { 
			int newProjectionNumber = numProjectionMatrices * numSweeps;
			Projection [] newMatrices = new Projection[newProjectionNumber];
			double [] newAngles = new double[newProjectionNumber];

			// Sweeps go now backward then forward ...
			for (int sweep = 0; sweep < numSweeps; sweep++) {
				int index = sweep * numProjectionMatrices;
				boolean forward = (sweep % 2 == 0);
				for (int i = 0; i < numProjectionMatrices; i++){
					if (forward) {
						newMatrices[index + i] = projectionMatrices[i];
						newAngles[index + i] = primaryAngles[i];
					} else { // backward
						newMatrices[index + numProjectionMatrices - i -1] = projectionMatrices[i];
						newAngles[index + numProjectionMatrices - i -1] = primaryAngles[i];
					}
				}	
			}
			projectionMatrices = newMatrices;
			numProjectionMatrices = newProjectionNumber;
			primaryAngles = newAngles;
		}
	}

	public static int getImageIndexInSingleSweepGeometry(int index){
		Configuration config =  Configuration.getGlobalConfiguration();
		int revan = index;
		if (config.getNumSweeps() > 1){
			int numProjectionMatrices = config.getGeometry().getNumProjectionMatrices();
			int numProjectionsPerSweep = numProjectionMatrices / config.getNumSweeps();
			int sweep = index / numProjectionsPerSweep;
			boolean forward = (sweep % 2 == 0);
			revan = index - (sweep * numProjectionsPerSweep);
			if (!forward) {
				revan = numProjectionsPerSweep - index - 1;
			} 
		}
		return revan;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
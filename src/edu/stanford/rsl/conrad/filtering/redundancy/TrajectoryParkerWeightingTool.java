package edu.stanford.rsl.conrad.filtering.redundancy;

import ij.IJ;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * This version of the Parker weights computes the primary angles from the trajectory rather than from preconfigured "primaryAngles".
 * The correct values are computed during the configure state. We use a gradient descent procedure the compute the iso-center.
 * Results are way more stable, but it takes a short while.
 * 
 * 
 * @author akmaier
 *
 */
public class TrajectoryParkerWeightingTool extends ParkerWeightingTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7809768945811552837L;

	@Override
	public void configure() throws Exception {
		Configuration config = Configuration.getGlobalConfiguration(); 
		setConfiguration(config);
		
		Trajectory trajectory = new Trajectory(config.getGeometry());
		// compute correct primary angles from projection matrices.
		trajectory.updatePrimaryAngles();
		setPrimaryAngles(trajectory.getPrimaryAngles());
		//double [] minmax = null;
		checkDelta();
		double maxRange = (Math.PI + (2 * delta));
		if (getPrimaryAngles() != null){
			//minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
			double range = computeScanRange() * Math.PI / 180;
			offset = (maxRange - range) /2;
			if (debug) CONRAD.log("delta: " + delta * 180 / Math.PI);
			if (debug) CONRAD.log("Angular Offset: " + offset * 180 / Math.PI + " " + maxRange + " " + range);
		}

		//offset = UserUtil.queryDouble("Offset for Parker weights: ", offset);
		setNumberOfProjections(config.getGeometry().getPrimaryAngles().length);
		if (this.numberOfProjections == 0){
			throw new Exception("Number of projections not known");
		}
		setConfigured(true);
	}
	

	@Override
	public String getToolName() {
		return "Parker Redundancy Weighting Filter (uses projection matrices)";
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
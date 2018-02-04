package edu.stanford.rsl.tutorial.atract;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Class to simulate a collimation on existing projection images.
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class Kollimator {

	private int maxThetaIndex;
	private int maxSIndex;

	/**
	 * This constructor is only necessary for 2-D Kollimation. The 3-D Collimation reads the necessary values directly from the Configuration file.
	 * @param maxThetaIndex Number of projections
	 * @param maxSIndex Detector size in pixels
	 */
	public Kollimator(int maxThetaIndex, int maxSIndex) {
		this.maxThetaIndex = maxThetaIndex;
		this.maxSIndex = maxSIndex;
	}

	public Kollimator() {
	}

	/**
	 * This function collimates an existing projection image stack. Pixels outside the kollimator are set to zero.
	 * The collimator is centered by default. Width is the still visible part of the projection provided by user input.
	 * @param input The 2-D sinogram
	 * @param width Kollimator width
	 */
	public void applyToGrid(Grid2D input, int width) {

		assert (width < maxSIndex);

		int kollCenter = maxSIndex / 2;
		for (int t = 0; t < maxThetaIndex; t++) {
			for (int s = 0; s < maxSIndex; s++) {
				if (s < kollCenter - width / 2 || s > kollCenter + width / 2)
					input.setAtIndex(s, t, 0);
			}
		}
	}

	/**
	 * This function kollimates an existing projection image stack. Pixels outside the kollimator are set to zero.
	 * The kollimator is centered by default. Width and height define the still visible field of view and are provided by user input.
	 * @param input	The projection images in a 3-D Grid
	 * @param width Kollimator width
	 * @param height Kollimator height
	 */
	public void applyToGrid(Grid3D input, int width, int height) {
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int maxV = geo.getDetectorHeight();
		int maxU = geo.getDetectorWidth();
		int maxProjs = conf.getGeometry().getProjectionStackSize();

		assert (width < maxU && height < maxV);

		for (int p = 0; p < maxProjs; p++) {
			int kollCenterV = maxV / 2;
			int kollCenterU = maxU / 2;

			for (int v = 0; v < maxV; v++) {
				for (int u = 0; u < maxU; u++) {
					if (v < kollCenterV - height / 2 || v > kollCenterV + height / 2 || u > kollCenterU + width / 2
							|| u < kollCenterU - width / 2)
						input.setAtIndex(u, v, p, 0.f);
				}
			}
		}

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
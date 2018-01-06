/*
 * Copyright (C) 2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.apps.gui.opengl;

import java.awt.Color;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Renders a scene with the currently configured source and detector pairs.
 * @author akmaier
 *
 */

public class TrajectoryViewer extends PointCloudViewer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5224405819368058599L;
	static int steps = 10;
	static boolean plotDetector = true;

	public TrajectoryViewer() {
		super("Current Trajectory", generatePoints());
		ArrayList<Color> colors = new ArrayList<Color>();

		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		for (int i=0; i < traj.getNumProjectionMatrices(); i++){
			int increment = (int) ((100.0 /traj.getNumProjectionMatrices()) * i);
			colors.add(new Color (205,205,105+increment));
			if (plotDetector){
				for (int v=0; v < steps; v++) {
					for (int u=0; u < steps; u++){
						colors.add(new Color (180,80+increment,0));					
					}
				}
			}
		}
		setColors(colors);
	}

	protected static ArrayList<PointND> generatePoints(){
		ArrayList<PointND> points = new ArrayList<PointND>();
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		int stepU = traj.getDetectorWidth()/steps;
		int stepV = traj.getDetectorHeight()/steps;
		for (int i=0; i < traj.getNumProjectionMatrices(); i++){
			Projection projection = traj.getProjectionMatrix(i);
			SimpleVector sourceVector = projection.computeCameraCenter();
			PointND source = new PointND(sourceVector);
			points.add(source);
			if (plotDetector){
				for (int v=0; v < steps; v++) {
					for (int u=0; u < steps; u++){
						SimpleVector coordinate = new SimpleVector (u*stepU,v*stepV);
						points.add(projection.computeDetectorPoint(sourceVector, coordinate, traj.getSourceToDetectorDistance(), traj.getPixelDimensionX(), traj.getPixelDimensionY(),  traj.getDetectorWidth(), traj.getDetectorHeight()));
					}
				}
			}
		}

		return points;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Configuration.loadConfiguration();
		TrajectoryViewer trv = new TrajectoryViewer();
		trv.setVisible(true);
	}

}

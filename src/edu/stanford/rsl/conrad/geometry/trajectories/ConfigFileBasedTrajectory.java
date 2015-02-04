package edu.stanford.rsl.conrad.geometry.trajectories;

import edu.stanford.rsl.conrad.io.ConfigFileParser;


/**
 * Abstract class to create config file-based geometry sources. This class and all of it's subclasses should
 * not implement the default constructor as they should not be XML-Serializable.
 * @author akmaier
 *
 */
public abstract class ConfigFileBasedTrajectory extends Trajectory implements ConfigFileParser {



	/**
	 * 
	 */
	private static final long serialVersionUID = -5999736125816034590L;

	public ConfigFileBasedTrajectory(Trajectory model) {
		super(model);
	}

	public static ConfigFileBasedTrajectory [] getGeometrySources(Trajectory model){
		ConfigFileBasedTrajectory [] sources = {new ProjectionTableFileTrajectory(model), 
				new SystemGeometryConfigFileTrajectory(model), new DennerleinProjectionTableFileTrajectory(model)};
		return sources;
	}

	public static Trajectory openAsGeometrySource(String filename, Trajectory geometry){
		System.out.println("Attempting to load: " + filename);
		ConfigFileBasedTrajectory [] sources = getGeometrySources(geometry);
		ConfigFileBasedTrajectory test = null;
		for (int i = 0; i < sources.length; i++){
			test = sources[i];
			try {
				test.readConfigFile(filename);
				break;
			} catch (Exception projectionTable){
				projectionTable.printStackTrace();
				System.out.println(projectionTable.getLocalizedMessage());
				test = null;
				//projectionTable.printStackTrace();
			}
		}
		System.out.println("Loaded trajectory with " + test.numProjectionMatrices + " matrices");
		return new Trajectory(test);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
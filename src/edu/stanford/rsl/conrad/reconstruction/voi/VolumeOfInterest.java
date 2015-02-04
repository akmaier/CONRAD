package edu.stanford.rsl.conrad.reconstruction.voi;

import java.io.Serializable;

import edu.stanford.rsl.conrad.io.ConfigFileParser;

/**
 * An abstract description of an arbitrary volume-of-interest (VOI).
 * @author akmaier
 *
 */
public abstract class VolumeOfInterest implements ConfigFileParser, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -9014453765525455074L;
	boolean debug = false;

	/**
	 * Tests whether a given coordinate in world coordinates is within the VOI.
	 * @param x world coordinate x
	 * @param y world coordinate y
	 * @param z world coordinate z
	 * @return true, if the coordinate is insider the VOI.
	 */
	public abstract boolean contains (double x, double y, double z);

	/**
	 * Reports a list of all known implementations of VolumeOfInterest.
	 * @return the list of implementations.
	 */
	public static VolumeOfInterest [] getVolumes(){
		VolumeOfInterest [] volumes = {new PolygonBasedVolumeOfInterest(), new CylinderBasedVolumeOfInterest()};
		return volumes;
	}

	/**
	 * Constructor from filename
	 * @param filename the filename
	 * @return the volume of interest.
	 */
	public static VolumeOfInterest openAsVolume(String filename) {
		VolumeOfInterest [] volumes = getVolumes();
		VolumeOfInterest revan = null;
		for (int i = 0; i < volumes.length; i++){
			revan = volumes[i];
			try {
				revan.readConfigFile(filename);
				break;
			} catch (Exception e) {
				System.out.println(e.getLocalizedMessage());
				revan = null;
				//e.printStackTrace();
			}
		}
		return revan;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2014 Michael Manhart, Kerstin Müller
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.tutorial.phantoms;


public class BrainPerfusionPhantomConfig {
	public final static int sizeX = 256;
	public final static int sizeY = 256;
	public final static int sizeZ = 256;
	public final static int size = sizeX*sizeY*sizeZ;
	public final static float voxelSize = 1.0f; // [mm]
	public final static float originX = 0.0f;
	public final static float originY = 0.0f;
	public final static float originZ = 0.0f;
	boolean motion = false;
		
	public String phantomDirectory;
	public float  phantomSampling; 

	public String calibrationFwdMatFile;
	public String calibrationBwdMatFile;
		
	public String phantomOutDirectory;
	
	// Should also be configurable later over the GUI
	// Default values for contrast flow
	// Start of rotation after contrast injection
	float tStart 	= 6.2f;
	// Duration of one C-arm sweep
	float tRot    	= 4.2f;
	// Duration of the pause before the backward sweep
	float tPause 	= 1.f;
	// Number of sweeps in total
	float numSweeps = 7;

	// xml file to 3D marker position
	public String markerFile = "";
}

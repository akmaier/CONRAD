package edu.stanford.rsl.conrad.pipeline;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.IndividualFilesProjectionDataSink;
import edu.stanford.rsl.conrad.io.SafeSerializable;
import edu.stanford.rsl.conrad.utils.Configuration;


public abstract class BufferedProjectionSink implements ProjectionSink, GUIConfigurable, Citeable, SafeSerializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8818811805620422888L;
	protected Grid3D projectionVolume;
	protected boolean configured = false;
	protected boolean showStatus = false;

	public void setShowStatus(boolean showStatus){
		this.showStatus = showStatus;
	}

	public abstract void process(Grid2D projection, int projectionNumber) throws Exception;
	public abstract String getName();
	
	public Grid3D getResult(){
		configured = false;
		adjustViewRange();
		return projectionVolume;
	}

	public Grid3D getProjectionVolume() {
		return projectionVolume;
	}

	public abstract void setConfiguration(Configuration config);

	public static BufferedProjectionSink [] getProjectionDataSinks (){
		BufferedProjectionSink [] sinks = { 	 
				new ImagePlusDataSink(),
				new IndividualFilesProjectionDataSink(),
		};
		return sinks;
	}
	
	public String toString(){
		return getName();
	}

	public void adjustViewRange(){

	}
	

	@Override
	public void prepareForSerialization(){
		configured = false;
		projectionVolume = null;
	}

	public boolean isConfigured() {
		return configured;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
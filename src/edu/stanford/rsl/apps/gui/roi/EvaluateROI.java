/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.gui.roi;

import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.utils.CONRAD;
import ij.ImagePlus;
import ij.gui.Roi;

public abstract class EvaluateROI implements GUIConfigurable {
	protected ImagePlus image = null;
	protected boolean configured = false;
	protected Roi roi = null;
	protected boolean debug = false;
	
	public ImagePlus getImage() {
		return image;
	}

	public void setImage(ImagePlus image) {
		this.image = image;
	}
	
	

	public EvaluateROI(){
		
	}
	
	public static EvaluateROI [] knownMethods(){
		ArrayList<Object> list = null;
		try {
			list = CONRAD.getInstancesFromConrad(EvaluateROI.class);
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		EvaluateROI [] known = new EvaluateROI [list.size()];
		list.toArray(known);
		return known;
	}
	
	public boolean isConfigured() {
		return configured;
	}

	
	public abstract Object evaluate();
	
	public abstract String toString();

	public Roi getRoi() {
		return roi;
	}

	public void setRoi(Roi roi) {
		this.roi = roi;
	}

	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
}



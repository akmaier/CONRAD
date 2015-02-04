/*
 * Copyright (C) 2014 Andreas Maier, Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import edu.stanford.rsl.conrad.physics.absorption.PolychromaticAbsorptionModel;

public class SimplePolychromaticDetector extends XRayDetector {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4294279623444760432L;

	public SimplePolychromaticDetector(){
		super();
		this.model = new PolychromaticAbsorptionModel();
	}
	
	@Override
	public void configure() throws Exception{
		model.configure();
	}
	
	public String toString(){
		String name = "Simple Polychromatic X-Ray Detector";
		if (model!= null) name+=" " + model.toString();
		return name;
	}
	
}

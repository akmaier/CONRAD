/*
 * Copyright (C) 2014 Andreas Maier, Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;


import edu.stanford.rsl.conrad.physics.absorption.SelectableEnergyMonochromaticAbsorptionModel;

public class SimpleMonochromaticDetector extends XRayDetector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3174333267925811334L;

	public SimpleMonochromaticDetector() {
		super();
		this.model =  new SelectableEnergyMonochromaticAbsorptionModel();
		((SelectableEnergyMonochromaticAbsorptionModel)model).configure(55);
	}
	
	@Override
	public void configure(){
		
	}
	
	public String toString(){
		String name = "Simple Monochromatic X-Ray Detector";
		if (model!= null) name+=" " + model.toString();
		return name;
	}
	
}

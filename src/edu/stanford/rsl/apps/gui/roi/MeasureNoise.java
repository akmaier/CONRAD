package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;

public class MeasureNoise extends EvaluateROI {

	
	@Override
	public Object evaluate() {
		if (configured) {
			image.setRoi(roi);
			double result = image.getChannelProcessor().getStatistics().stdDev;
			System.out.println("Roi statistics: " + image.getChannelProcessor().getStatistics().toString());
			System.out.println("Standard Deviation: " + result);
		}
		return null;
	}
	
	public void configure() throws Exception {
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure standard deviation of an roi";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

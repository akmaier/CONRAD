
import ij.*;
import ij.measure.Measurements;
import ij.plugin.PlugIn;
import ij.plugin.filter.Analyzer;


/** This plugin Measures all slices in the current ImagePlus. 
	 */
public class Measure_All_Slices implements PlugIn {

	public Measure_All_Slices(){
		
	}

	public void run(String arg) {
		ImagePlus current = IJ.getImage();
		Analyzer.setMeasurement(Measurements.LABELS, true);
		for (int i=0; i<current.getStackSize(); i++) {
			current.setSlice(i+1);
			IJ.showProgress(i+1, current.getStackSize());
			IJ.run(current, "Measure", "");	
		}
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
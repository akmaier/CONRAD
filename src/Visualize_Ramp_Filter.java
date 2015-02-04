import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.filtering.rampfilters.*;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.plugin.PlugIn;


public class Visualize_Ramp_Filter implements PlugIn {

	public Visualize_Ramp_Filter(){
		
	}
	
	@Override
	public void run(String arg) {
		Configuration.loadConfiguration();
		RampFilter [] ramps = RampFilter.getAvailableRamps();
		for (RampFilter ramp : ramps){
			ramp.setConfiguration(Configuration.getGlobalConfiguration());
		}
		RampFilter ramp = (RampFilter) JOptionPane.showInputDialog(null, "Please select the Ramp Filter", "Ramp Filter Selection", JOptionPane.DEFAULT_OPTION, null, ramps, ramps[0]);
		try {
			ramp.configure();
			VisualizationUtil.createComplexPowerPlot(ramp.getRampFilter1D(1024)).show();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
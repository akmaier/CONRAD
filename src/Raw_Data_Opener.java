import edu.stanford.rsl.apps.gui.RawDataOpener;
import edu.stanford.rsl.conrad.utils.Configuration;

import ij.plugin.PlugIn;


public class Raw_Data_Opener implements PlugIn {

	public Raw_Data_Opener(){
	}
	
	@Override
	public void run(String arg) {
		Configuration.loadConfiguration();
		new RawDataOpener().setVisible(true);
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
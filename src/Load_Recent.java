import java.io.File;

import edu.stanford.rsl.conrad.io.NrrdFileReader;
import edu.stanford.rsl.conrad.utils.Configuration;
import ij.plugin.PlugIn;


public class Load_Recent implements PlugIn {

	public Load_Recent(){
		
	}
	
	@Override
	public void run(String arg) {
		Configuration.loadConfiguration();
		NrrdFileReader reader = new NrrdFileReader();
		File file = new File(Configuration.getGlobalConfiguration().getRecentFileOne());
		reader.load(file.getParentFile().getAbsolutePath(), file.getName()).show();
		reader = new NrrdFileReader();
		file = new File(Configuration.getGlobalConfiguration().getRecentFileTwo());
		reader.load(file.getParentFile().getAbsolutePath(), file.getName()).show();
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
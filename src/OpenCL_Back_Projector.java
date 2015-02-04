import ij.IJ;
import ij.plugin.PlugIn;
import edu.stanford.rsl.conrad.opencl.OpenCLBackProjector;
import edu.stanford.rsl.conrad.utils.Configuration;


public class OpenCL_Back_Projector implements PlugIn {

	/**
	 * Required by ImageJ
	 */
	public OpenCL_Back_Projector(){

	}

	@Override
	public void run(String arg) {
		Configuration.loadConfiguration();
		OpenCLBackProjector bp = new OpenCLBackProjector();
		bp.setConfiguration(Configuration.getGlobalConfiguration());
		String name = IJ.getImage().getTitle();
		try {
			bp.setShowStatus(true);
			bp.reconstructOffline(IJ.getImage());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

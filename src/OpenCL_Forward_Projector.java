import edu.stanford.rsl.conrad.opencl.OpenCLForwardProjector;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;


public class OpenCL_Forward_Projector implements PlugIn {

	public OpenCL_Forward_Projector(){
		
	}
	
	@Override
	public void run(String arg) {
		// TODO Auto-generated method stub
		OpenCLForwardProjector projector = new OpenCLForwardProjector();
		try {
			projector.configure();
			projector.setTex3D(IJ.getImage());
			ImagePlus projections = projector.project();
			projections.show();
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
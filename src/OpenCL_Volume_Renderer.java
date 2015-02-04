import edu.stanford.rsl.conrad.opencl.rendering.OpenCLVolumeRenderer;
import ij.IJ;
import ij.plugin.PlugIn;

/**
 * Creates a OpenCL Volume Rendering.
 * 
 * @author Martin Berger
 *
 */
public class OpenCL_Volume_Renderer implements PlugIn {

	public OpenCL_Volume_Renderer(){
		
	}
	
	@Override
	public void run(String arg) {
		OpenCLVolumeRenderer renderer = new OpenCLVolumeRenderer(IJ.getImage());
		renderer.start();
	}

}

/*
 * Copyright (C) 2010-2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
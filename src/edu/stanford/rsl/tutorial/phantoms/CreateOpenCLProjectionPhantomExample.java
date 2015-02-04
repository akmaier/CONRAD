/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.phantoms;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.opencl.OpenCLProjectionPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Example to create a stack of projections using the API.
 * 
 * @author akmaier
 */
public class CreateOpenCLProjectionPhantomExample {

	static PhantomRenderer phantom;
	
	public static void main(String[] args) {
		new ImageJ();
		Configuration.loadConfiguration();
		try {
			phantom = new OpenCLProjectionPhantomRenderer();
			phantom.configure();
			Grid3D projections = PhantomRenderer.generateProjections(phantom);
			projections.show(phantom.toString());
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
			
	}

}

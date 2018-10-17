package edu.stanford.rsl.tutorial.basics.videoTutorials;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.ImageJ;
import ij.IJ;
import ij.ImagePlus;

public class videoTutorialsImageJ {
	public static void main(String[] args) {
		
		String imgPath = System.getProperty("user.dir") + "/data" + "/mipda/";
		String filename = imgPath + "vessel.jpg";
		String outFilename = imgPath + "output/" + "vessel_new.tiff";
		
 		// open the ImageJ GUI
		ImageJ imagej = new ImageJ();
		
		// open an image using an ImagePlus instance and show it
		ImagePlus imp = IJ.openImage(filename);
		imp.show();
		
 		
		// Running ImageJ commands on an image
		// Making a binary image  
		IJ.run(imp, "Convert to Mask", ""); 
		
		
		// Gaussian blur the image
		IJ.run(imp, "Gaussian Blur...", "sigma=1.5");
 		
		
		
		// Wrap an ImagePlus instance to Grid2D
		Grid2D img = ImageUtil.wrapImagePlus(
				IJ.openImage(filename)).getSubGrid(0);
		img.show("Grid image");
		
		
 		// Wrap a Grid2D instance back to ImagePlus
		ImagePlus imgp = new ImagePlus(
				"Image", ImageUtil.wrapGrid2D(img).createImage());
		IJ.saveAsTiff(imgp, outFilename);
	
	}
}

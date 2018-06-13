package edu.stanford.rsl.tutorial.mipda;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleVector.VectorNormType;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.ImageJ;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.process.FloatProcessor;
import ij.gui.Roi;

public class videoTutorialsImageJ {
public static void main(String[] args) {

		String filename = "C:/vessel.jpg";
		String outFilename = "vessel_new.tiff";
		

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

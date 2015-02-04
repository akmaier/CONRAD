
import java.io.File;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.io.OpenDialog;
import ij.measure.ResultsTable;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

/**
 * 
 * ImageJ Plugin to compute the number of edge pixels given a threshold.
 * Edges are extracted with the Sobel operator in x and y direction.
 * Then the edge magnitude is computed as the magnitude of the directional vector of each pixel.
 * 
 * Pixels and edge pixels are counted and the result is displayed for a complete directory.
 * 
 * This software is provided as is and may be redistributed freely with modifications as long as this comment 
 * is preserved.
 * 
 * @author Andreas Maier
 *
 */
public class Count_Edge_Points implements PlugIn {

	@Override
	public void run(String arg) {
		ij.io.OpenDialog od = new OpenDialog("Compute Edge Count - Directory Selection", IJ.getDirectory("image"), "*.*");
		if(od.getDirectory()!=null){
			String dir = od.getDirectory();   
			if (null == dir) return; // user canceled dialog
			ij.gui.GenericDialog gd = new GenericDialog("Threshold for edge detection");
			gd.addNumericField("Threshold", 128, 3);
			gd.showDialog();
			if (gd.wasCanceled()){
				throw new RuntimeException("User cancelled selection.");
			} 
			double threshold = gd.getNextNumber();
			File file = new File(dir);
			ij.measure.ResultsTable table = new ResultsTable();
			for (String fString: file.list()){
				ImagePlus image = IJ.openImage(dir+"/"+fString);
				table.incrementCounter();
				table.addLabel(fString);
				if (image != null){
					System.out.println(fString);
					ImageProcessor iProcessor = image.getProcessor();
					ImageProcessor floatP = iProcessor.convertToFloat();
					ImageProcessor floatP2 = iProcessor.convertToFloat();
					floatP.convolve3x3(new int [] {-1, -1, -1, 0, 0, 0, 1, 1, 1});
					floatP2.convolve3x3(new int [] {-1, 0, 1, -1, 0, 1, -1, 0, 1});
					floatP.sqr();
					floatP2.sqr();
					int edge = 0;
					int count = floatP.getHeight() * floatP.getWidth();
					for (int i = 0; i < count; i++){
						double value = Math.sqrt(((float []) floatP.getPixels())[i] + ((float []) floatP2.getPixels())[i]);
						if (value > threshold) edge ++;
					}
					table.addValue("Image Area", count);
					table.addValue("Edge Points", edge);
					table.addValue("Percentage", ((double) edge * 100.0) / count);
				}

			}
			table.show("Edge Measurements");

		}

	}

}


/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
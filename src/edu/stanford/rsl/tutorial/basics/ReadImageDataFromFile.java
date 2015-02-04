/*
 * Copyright (C) 2014 Andreas Maier, Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.basics;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;


/**
 * This is a simple tutorial on how to use ImageJ for reading files from hard disk to CONRAD's Grid containers.
 * 
 * @author akm
 *
 */
public class ReadImageDataFromFile {

	/**
	 * Main routine for this example.
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			// we need ImageJ in the following
			new ImageJ();
			// locate the file
			// here we only want to select files ending with ".bin". This will open them as "Dennerlein" format.
			// Any other ImageJ compatible file type is also OK.
			// new formats can be added to HandleExtraFileTypes.java
			String filenameString = FileUtil.myFileChoose(".bin", false);
			// call the ImageJ routine to open the image:
			ImagePlus imp = IJ.openImage(filenameString);
			// Convert from ImageJ to Grid3D. Note that no data is copied here. The ImageJ container is only wrapped. Changes to the Grid will also affect the ImageJ ImagePlus.
			Grid3D impAsGrid = ImageUtil.wrapImagePlus(imp);
			// Display the data that was read from the file.
			impAsGrid.show("Data from file");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

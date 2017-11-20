/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.basics;

import java.io.File;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.SammonObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.utils.HelperClass;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.io.DirectoryChooser;

/**
 * Simple example on how to create a Sammon map from a directory containing image files. Dimension can either be set 2-D or 3-D
 * 
 * @author maier
 *
 */
public class SammonMappingExample {

	public static void main(String[] args) {
		boolean writeMap = true;
		int dimension = 3;
		DirectoryChooser pickDir = new DirectoryChooser("Pick a directory");
		String imagePath = pickDir.getDirectory();
		File file = new File (imagePath);
		ArrayList<NumericGrid> imageList = new ArrayList<NumericGrid>();
		ArrayList<String> fileList = new ArrayList<String>();
		// Read data from disk
		for (String imageFile : file.list()){
			if (imageFile.endsWith("png") || imageFile.endsWith("tif") || imageFile.endsWith("jpg")  || imageFile.endsWith("dcm") ){
				//Read image data
				Grid2D gridImage = ImageUtil.wrapImageProcessor(IJ.openImage(imagePath+"/"+imageFile).getChannelProcessor());
				// add to lists
				imageList.add(gridImage);
				fileList.add(imageFile);
			}
		}
		// Get a dimensionality reduction for overview: Here we use the sammon mapping
		DimensionalityReduction dimRed = new DimensionalityReduction(HelperClass.buildDistanceMatrix(imageList)); 
		PointCloudViewableOptimizableFunction gradFunc = new SammonObjectiveFunction(); 
		dimRed.setTargetFunction(gradFunc); 
		dimRed.setTargetDimension(dimension);
		// set the argument of optimize to true to get an on-the-fly visualization
		double [] points = dimRed.optimize(true, 1897238234233747l);
		if (writeMap) {
			for (int i=0; i <fileList.size(); i++){
				// write map to stdout for plotting if "writeMap" is set
				System.out.print(fileList.get(i));
				for (int j =0; j< dimension;j++) System.out.print(" " + points[i*dimension+j]);
				System.out.print("\n");
			}
		}

	}

}

/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics;

import java.io.File;
import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;

public class XRayAnalytics {
	
	//Constants used in SSIM. Can be changed
	private final static double k1 = 0.01;
	private final static double k2 = 0.03;
	
	/**
	 * Uses the method System.getProperty("user.dir") to access actual directory
	 * @return returns parent directory based on the CONRAD setup
	 */
	public static String getPathToDir(){
		String path = System.getProperty("user.dir");
		//Working with the Conrad project in an eclipse workspace
		try {
			path = new File(System.getProperty("user.dir") + "../../..").getCanonicalPath(); // /Results
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		path += "/";
		
		return path;
	}
	
	/**
	 * @param title = Name of the file to open. Can contain directory path.
	 * @return Opened file as a Grid2D
	 */
	public static Grid2D openFile(String title) {
		
		String path = getPathToDir() + title;
		if(! title.substring(title.length()-4).equals(".tif") ) {
			//System.err.println("File doesnt end with .tif!");
			path+=".tif";
		}

		Grid3D roi = ImageUtil.wrapImagePlus(IJ.openImage(path));
		Grid2D grid = roi.getSubGrid(0);
		
		return grid;
	}
	
	/**
	 * @param grid = result that should be saved
	 * @param title = Name of the file to save. Can contain directory path.
	 */
	public static void saveFile(Grid2D grid, String title){
		if (title==null || title.equals("")){
			System.err.println("A file is not now allowed to have an empty title!");
			return;
		}
		
		String path = getPathToDir() + title;		
		
		System.out.println("Saving file to: " + path);
		
		ImagePlus dummy = new ImagePlus(title, ImageUtil.wrapGrid2D(grid));
		if(!IJ.saveAsTiff(dummy, path))
			System.err.println("Error while saving file as tiff!");
		
		return;
	}
	
	/**
	 * Visual output of the given grid
	 */
	public static void showGrid2D(Grid2D grid, String title, double min, double max){	
		ImagePlus imp = new ImagePlus();
		ImageProcessor image = ImageUtil.wrapGrid2D(grid);
		// Set the range of the image!
		if(min != 0.0 || max != 0.0)
			image.setMinAndMax(min, max);
		
		ImageStack stack = new ImageStack(image.getWidth(), image.getHeight());
		stack.addSlice(title, image);
		imp.setStack(title, stack);
		imp.show();
	}
	
	/**
	 * Compares two files with the RMSE Method. (root-mean-square deviation (RMSD) or root-mean-square error (RMSE))
	 * @param first FileName
	 * @param second FileName
	 * @return RMSD is always non-negative, and a value of 0 (almost never achieved in practice) would indicate a perfect fit to the data. In general, a lower RMSD is better than a higher one.
	 */
	public static double computeRMSE(String firstFileName, String secondFileName){
		Grid2D first = openFile(firstFileName);
		Grid2D second = openFile(secondFileName);
		 
		return computeRMSE(first, second);
	}
	
	/**
	 * Compares two files with the RMSE Method. (root-mean-square deviation (RMSD) or root-mean-square error (RMSE))
	 * @param first grid
	 * @param second grid
	 * @return RMSD is always non-negative, and a value of 0 (almost never achieved in practice) would indicate a perfect fit to the data. In general, a lower RMSD is better than a higher one.
	 */
	public static double computeRMSE(Grid2D first, Grid2D second){
		if(first.getWidth() != second.getWidth() || first.getHeight() != second.getHeight()){
			System.err.println("The grids in \"totalDifference\" do not have the same dimensions");
			return -1;
		}
		
		return nHelperFkt.computeRMSE(first, second);
	}
	
	public static double averageRMSE(String firstFileName, String secondFileName){
		Grid2D first = openFile(firstFileName);
		Grid2D second = openFile(secondFileName);
		
		return averageRMSE(first, second);
	}
	
	public static double averageRMSE(Grid2D first, Grid2D second){
		if(first.getWidth() != second.getWidth() || first.getHeight() != second.getHeight()){
			System.err.println("The grids in \"totalDifference\" do not have the same dimensions");
			return -1;
		}
		
		Grid2D a = nHelperFkt.compareToAveragePixel(first);
		Grid2D b = nHelperFkt.compareToAveragePixel(second);
		
		return nHelperFkt.computeRMSE(a, b);
	}
	
	/**
	 * Determines the absolute difference between the individual pixels of two grids
	 * @param first grid
	 * @param second grid
	 * @return Sum of the absolut difference
	 */
	public static double totalDifference(Grid2D first, Grid2D second) {
		if(first.getWidth() != second.getWidth() || first.getHeight() != second.getHeight()){
			System.err.println("The grids in \"totalDifference\" do not have the same dimensions");
			return -1;
		}
		
		double total = 0.0;
		for (int y = 0; y < first.getHeight(); y++) {
			for (int x = 0; x < first.getWidth(); x++) {
//				if (first.getAtIndex(x, y) != 0.0) {
//					//System.out.println("Compare: " + grid.getAtIndex(x, y) + " with " + groundTruth.getAtIndex(x, y));
//				}
				double diff = first.getAtIndex(x, y) - second.getAtIndex(x, y);
				total += Math.abs(diff);
			}
		}
		return total;
	}
	
	
	/**
	 * Creates a new grid where every value of the original grid is divided by the given float. Used to adjust the results to energy per ray.
	 * @param grid
	 * @param divider
	 * @return new grid with resulting values
	 */
	public static Grid2D divideValue(Grid2D grid, float divider) {
		Grid2D result = new Grid2D(grid.getWidth(), grid.getHeight());
		
		for (int y = 0; y < grid.getHeight(); y++) {
			for (int x = 0; x < grid.getWidth(); x++) {
				result.addAtIndex(x, y, (grid.getAtIndex(x, y) / divider));
			}
		}
			
		return result;
	}
	
	/**
	 * Testing function to visualize the application in the output!
	 */
	public static Grid2D halfAddValue(Grid2D grid, float adder) {
		Grid2D result = new Grid2D(grid.getWidth(), grid.getHeight());
		
		for (int y = 0; y < grid.getHeight(); y++) {
			for (int x = 0; x < grid.getWidth() / 2; x++) {
				result.addAtIndex(x, y, (grid.getAtIndex(x, y) + adder));
			}
			for (int x = grid.getWidth() / 2; x < grid.getWidth(); x++) {
				result.addAtIndex(x, y, (grid.getAtIndex(x, y)));
			}
		}
			
		return result;
	}
	
	/**
	 * Creates a new grid with a constant offset added to the orignal
	 */
	public static Grid2D addValue(Grid2D grid, float adder) {
		
		Grid2D result = new Grid2D(grid.getWidth(), grid.getHeight());
		
		for (int y = 0; y < grid.getHeight(); y++) {
			for (int x = 0; x < grid.getWidth(); x++) {
				result.addAtIndex(x, y, (grid.getAtIndex(x, y) + adder));
			}
		}
			
		return result;
	}
	
	/**
	 * Create a new grid, composed of the two committed grids. Example: Add direct and indirect lighting together for final result.
	 * @param first grid
	 * @param second grid
	 * @return 
	 */
	public static Grid2D addGrid(Grid2D grid0, Grid2D grid1) {
		if(grid0.getWidth() != grid1.getWidth() || grid0.getHeight() != grid1.getHeight()){
			System.err.println("The grids in \"addGrid\" do not have the same dimensions");
			return null;
		}
		
		Grid2D result = new Grid2D(grid0.getWidth(), grid0.getHeight());
		
		for (int y = 0; y < grid0.getHeight(); y++) {
			for (int x = 0; x < grid0.getWidth(); x++) {
				result.addAtIndex(x, y, (grid0.getAtIndex(x, y) + (grid1.getAtIndex(x, y))));
			}
		}
			
		return result;
	
	}
	
	/**
	 * Total amount of enery on the grid
	 * @param grid
	 * @return Sum of all energy values in the given grid
	 */
	public static double totalEnergyCount(Grid2D grid) {
		double sum = 0.0;
		
		for (int y = 0; y < grid.getHeight(); y++) {
			for (int x = 0; x < grid.getWidth(); x++) {
				sum += grid.getAtIndex(x, y);
			}
		}
		
		return sum;
	}
	
	
	/**
	 * The structural similarity (SSIM) index is a method for predicting the perceived quality of digital images
	 * @param range = dynamic range of the pixel-values; max-min values for the intensity
	 * @param windowSize = the measure between two windows x and y of common size N×N (windowSize) 
	 * @param stepSize = windows are shifted by stepSize for each iteration
	 * @return SSIM index is a decimal value between -1 and 1, and value 1 is only reachable in the case of two identical sets of data
	 */
	public static double structuralSIM(Grid2D first, Grid2D second, double range, int windowSize, int stepSize) {
		
//		System.out.println("Starting SSIM Calculation | Range " + range + " | windowSize " + windowSize + " | stepSize " + stepSize);
		
		double ssim = 0.0;
		double numPixels = windowSize * windowSize;
		int numberOfWindows = 0;
		
		double width = first.getWidth();
		double height = first.getHeight();
		
		for (int j = 0; j < height - windowSize; j += stepSize) {// height - windowSize
			for (int i = 0; i < width - windowSize; i += stepSize) { // width - windowSize

				// Average of first, second
				double averageFirst = 0.0;
				double averageSecond = 0.0;
				
				for (int y = 0; y < windowSize; ++y) {
					for (int x = 0; x < windowSize; ++x) {
						averageFirst += first.getAtIndex(x+i, y+j);
						averageSecond += second.getAtIndex(x+i, y+j);
					}
				}
				averageFirst /= numPixels;
				averageSecond /= numPixels;
				
				double varianceFirst = 0.0;
				double varianceSecond = 0.0;
				double covariance = 0.0;
				
				for (int y = 0; y < windowSize; ++y) {
					for (int x = 0; x < windowSize; ++x) {
						varianceFirst 	+= Math.pow((first.getAtIndex(x+i, y+j) - averageFirst), 2);
						varianceSecond 	+= Math.pow((second.getAtIndex(x+i, y+j) - averageSecond), 2);
						covariance 		+= ((first.getAtIndex(x+i, y+j) - averageFirst) * (second.getAtIndex(x+i, y+j) - averageSecond));
					}
				}
				varianceFirst = Math.sqrt(varianceFirst);
				varianceSecond = Math.sqrt(varianceSecond);
				
				double c1 = Math.pow(k1 * range, 2);
				double c2 = Math.pow(k2 * range, 2);

				double numerator = (2 * averageFirst * averageSecond + c1) * (2 * covariance + c2);
				double denominator = (Math.pow(averageFirst, 2) + Math.pow(averageSecond, 2) + c1)* (Math.pow(varianceFirst, 2) + Math.pow(varianceSecond, 2) + c2);

				ssim += numerator / denominator;
				numberOfWindows++;
			}
		}
		
		return ssim / numberOfWindows;
	}
}


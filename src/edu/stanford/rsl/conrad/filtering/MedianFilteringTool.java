package edu.stanford.rsl.conrad.filtering;

import java.util.Arrays;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;


/**
 * Implements a simple median filter based on sorting of all values in the kernel.
 * The kernel is specified by its width and its height. The values on the border of the image
 * that cannot be computed by the kernel are set to 0 in the resulting image.
 * 
 * 
 * @author Happy Coding Seminar
 *
 */
public class MedianFilteringTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2206224407501839982L;
	private int kernelWidth = 1;
	private int kernelHeight = 1;
	
	/**
	 * Method to create the median filtered image. Image is filtered according to the internal parameters
	 * kernelWidth and kernelHeight by sorting and selecting the center entry of the kernel.
	 * @param input
	 * @return the median filtered image.
	 */
	public Grid2D getMedianFilteredImage(Grid2D input){
		//ImageProcessor output = input.convertToFloat();
		Grid2D output = new Grid2D(input);
		int borderX = (kernelWidth-1) / 2;
		int borderY = (kernelHeight-1) / 2;
		float [] kernel = new float [kernelWidth*kernelHeight];
		for (int j=borderY; j < output.getHeight() - borderY; j++){
			for (int i=borderX; i < output.getWidth() - borderX; i++){
				int index = 0;
				for (int v = j - borderY; v <= j + borderY; v++){
					for (int u = i - borderX; u <= i + borderX; u++){
						kernel[index] = input.getPixelValue(u, v);
						index++;
					}
				}
				Arrays.sort(kernel);
				output.putPixelValue(i, j, kernel[(kernelWidth*kernelHeight-1) / 2]);
			}
		}		
		
		return output;
	}
	
	
	@Override
	public Grid2D applyToolToImage(Grid2D input) {
		return getMedianFilteredImage(input);
	}
	
	

	@Override
	public IndividualImageFilteringTool clone() {
		MedianFilteringTool clone = new MedianFilteringTool();
		clone.setKernelHeight(kernelHeight);
		clone.setKernelWidth(kernelWidth);
		clone.setConfigured(configured);
		return clone;
	}

	public int getKernelWidth() {
		return kernelWidth;
	}

	public void setKernelWidth(int kernelWidth) {
		this.kernelWidth = kernelWidth;
	}

	public int getKernelHeight() {
		return kernelHeight;
	}

	public void setKernelHeight(int kernelHeight) {
		this.kernelHeight = kernelHeight;
	}

	@Override
	public String getToolName() {
		if (isConfigured()){
			return "Median Filtering Tool (Kernel " + kernelWidth + "x" + kernelHeight + ")";
		} else {
			return "Median Filtering Tool";
		}
	}

	@Override
	public void configure() throws Exception {
		int value = (int) Math.round(Double.parseDouble(JOptionPane.showInputDialog("Enter filter size: ", kernelWidth)));
		if (value % 2 != 1) value++;
		kernelWidth = value;
		kernelHeight = value;
		setConfigured(true);
	}
	
	public void configure(int kernelWidth, int kernelHeight){
		this.kernelHeight = kernelHeight;
		this.kernelWidth = kernelWidth;
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@BOOK{Paulus03-APR,\n" +
		"  author = {{Paulus}, D. W. R. and {Hornegger}, J.},\n" +
		"  title = {{Applied Pattern Recognition}},\n" +
		"  publisher = {GWV-Vieweg},\n" +
		"  address = {Wiesbaden, Germany},\n" +
		"  edition = {4th},\n" +
		"  year = {2003}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Paulus DWR, Hornegger J, Applied Pattern Recognition, 4th edition, GWV-Vieweg, Wiesbaden, Germany, 2003.";
	}

	/**
	 * is an image filter for noise reduction and hence not device dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.filtering;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class MeanFilteringTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5822100894598421376L;
	private float[] kernel;
	private int kernelWidth = 1;
	private int kernelHeight = 1;
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		ImageProcessor duplicateImageProcessor  = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		duplicateImageProcessor.setPixels(imageProcessor.getBuffer());
		duplicateImageProcessor.convolve(kernel, kernelWidth, kernelHeight);
		return imageProcessor;
	}
	
	public static float [] createKernel(int kernelWidth, int kernelHeight){
		int values = kernelWidth * kernelHeight;
		float [] kernel = new float[values];
		for (int i = 0; i< values; i++){
			kernel[i] = (float) 1.0 / values;
		}
		return kernel;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		MeanFilteringTool clone = new MeanFilteringTool();
		clone.setKernel(kernel);
		clone.setKernelHeight(kernelHeight);
		clone.setKernelWidth(kernelWidth);
		clone.setConfigured(configured);
		return clone;
	}

	public float[] getKernel() {
		return kernel;
	}

	public void setKernel(float[] kernel) {
		this.kernel = kernel;
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
			return "Mean Filtering Tool (Kernel " + kernelWidth + "x" + kernelHeight + ")";
		} else {
			return "Mean Filtering Tool";
		}
	}

	@Override
	public void configure() throws Exception {
		int value = (int) Math.round(Double.parseDouble(JOptionPane.showInputDialog("Enter filter size: ", kernelWidth)));
		if (value % 2 != 1) value++;
		kernelWidth = value;
		kernelHeight = value;
		kernel = createKernel(kernelWidth, kernelHeight);
		setConfigured(true);
	}
	
	public void configure(int kernelWidth, int kernelHeight){
		this.kernelHeight = kernelHeight;
		this.kernelWidth = kernelWidth;
		kernel = createKernel(kernelWidth, kernelHeight);
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

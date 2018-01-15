/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.opencl;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Tool for OpenCL binarization of a Grid 2D.
 * This tool uses the CONRAD internal Grid 2-D data structure.
 * 
 * @author Jennifer Maier
 *
 */
public class Binarization2DTool extends OpenCLFilteringTool2D {

	private static final long serialVersionUID = 5508773057260674367L;
	private double thres = 0.5;
	
	public Binarization2DTool() {
		this.kernelName = kernelname.BINARIZATION_2D;
	}
	
	@Override
	public void configure() throws Exception {		

		this.thres = (float) UserUtil.queryDouble("Enter threshold for binarization", thres);
		configured = true;

	}
	
	// Getter and Setter
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
	public double getThres() {
		return thres;
	}

	public void setThres(double thres) {
		this.thres = thres;
	}
	
	@Override
	protected void configureKernel() {
		filterKernel = program.createCLKernel("binarization2D");

		filterKernel.setForce32BitArgs(true);
		
		filterKernel
		.putArg(image)
		.putArg(result)
		.putArg(width)
		.putArg(height)
		.putArg(thres);	

	}

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getToolName() {
		return "OpenCL Binarization 2D";
	}

	@Override
	public ImageFilteringTool clone() {
		Binarization2DTool clone = new Binarization2DTool();
		clone.setThres(this.getThres());
		clone.setConfigured(this.configured);
		return clone;
	}
	
}

/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
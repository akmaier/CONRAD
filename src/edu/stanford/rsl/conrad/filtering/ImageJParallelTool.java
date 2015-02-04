package edu.stanford.rsl.conrad.filtering;

import ij.IJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Filtering tool to apply an arbitrary ImageJ operation in parallel.
 * 
 * @author Martin Berger
 *
 */
public class ImageJParallelTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4269226968123355343L;
	String operation = null;
	String parameters = null;
	
	
	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		ImageJParallelTool filter = new ImageJParallelTool();
		filter.operation = operation;
		filter.parameters = parameters;
		filter.configured = configured;
		return filter;
	}

	@Override
	public String getToolName() {
		return "ImageJ filtering tool";
	}

	public void setOperation(String operation) {
		this.operation = operation;
	}

	public String getOperation() {
		return operation;
	}
	
	public void setParameters(String parameters) {
		this.parameters = parameters;
	}

	public String getParameters() {
		return operation;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		ImagePlus ip = new ImagePlus("Slice Nr: " + this.getImageIndex(), ImageUtil.wrapGrid2D(imageProcessor));
		IJ.run(ip, operation, parameters);
		return ImageUtil.wrapImageProcessor(ip.getProcessor());
	}

	@Override
	public void configure() throws Exception {
		operation = UserUtil.queryString("Enter operation to be performed", "Gaussian Blur...");
		parameters = UserUtil.queryString("Enter parameters for operation: " + operation, "sigma=2");
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public void prepareForSerialization() {
		configured = false;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}
	
}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

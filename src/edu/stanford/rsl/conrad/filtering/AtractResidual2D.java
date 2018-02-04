package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.tutorial.atract.AtractKernel2D;


/**
 * Filters all projections of a stack with the 2D residual kernel of the ATRACT reconstruction
 * 
 * @author Martin Berger / Marco Boegel
 *
 */
public class AtractResidual2D extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5015484026254715749L;
	private double zeroValue = 3;
	private AtractKernel2D kernel = null;


	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		AtractResidual2D filter = new AtractResidual2D();
		filter.zeroValue = zeroValue;
		filter.configured = configured;
		filter.kernel = kernel;
		return filter;
	}

	@Override
	public String getToolName() {
		return "ATRACT-2D Residual Filter";
	}

	public void setZeroValue(double zeroValue) {
		this.zeroValue = zeroValue;
	}

	public double getZeroValue() {
		return zeroValue;
	}

	private void initialize(int x, int y){
		kernel = new AtractKernel2D(x, y, (float)zeroValue);
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		if (kernel == null){
			initialize(imageProcessor.getWidth(), imageProcessor.getHeight());
		}
		kernel.applyToGrid(imageProcessor);
		return imageProcessor;
	}

	@Override
	public void configure() throws Exception {
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return "@article{Dennerlein13-ATR,\n" +
		"number={17},\n" +
		"author={Frank Dennerlein and Andreas Maier},\n" +
		"keywords={trunction correction, computed tomography},\n" +
		"doi={10.1088/0031-9155/58/17/6133},\n" +
		"journal={Physics in Medicine and Biology},\n" +
		"volume={58},\n" +
		"title={{Approximate truncation robust computed tomography - ATRACT}},\n" +
		"year={2013},\n" +
		"pages={6133--6148}\n" +
		"}";
	}

	@Override
	public String getMedlineCitation() {
		return "Dennerlein F, Maier A. Approximate truncation robust computed tomography - ATRACT. Physics in Medicine and Biology, vol. 58, no. 17, pp. 6133-6148, 2013.";
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
 * Copyright (C) 2010-2014 - Marco Boegel, Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

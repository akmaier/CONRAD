package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class LaplaceFilteringTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -741259606020972883L;
	protected boolean twoD = false;
	
	public LaplaceFilteringTool (){
		configured = true;
	}
	
	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		LaplaceFilteringTool filter = new LaplaceFilteringTool();
		filter.configured = configured;
		filter.twoD = twoD;
		return filter;
	}

	
	@Override
	public void configure() throws Exception {
		twoD = UserUtil.queryBoolean("Compute in both directions?");
		configured = true;
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

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor)
			throws Exception {
		ImageProcessor revan = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		revan.setPixels(imageProcessor.getBuffer());
		if (!twoD){
			float kernel [] = {1,-2,1};
			revan.convolve(kernel, 3, 1);
			//revan.multiply(0.5/Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX());
		} else {
			float kernel [] = {0,1,0,1,-4,1,0,1,0};
			revan.convolve(kernel, 3, 3);
		}
		return imageProcessor;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		return "Laplace Filtering Tool";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

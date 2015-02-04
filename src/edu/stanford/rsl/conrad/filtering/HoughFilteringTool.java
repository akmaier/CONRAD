package edu.stanford.rsl.conrad.filtering;

import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.hough.FixedCircleHoughSpace;


/**
 * Tool to apply a Hough filter for circles with fixed dimension.
 * 
 * @author Andreas Maier
 *
 */
public class HoughFilteringTool extends IndividualImageFilteringTool {


	private static final long serialVersionUID = 1L;
	// radius of circles in pixels
	private int radiusOfCircles = 5;
	private double start = 40;
	private double stop = 55;
	
	public HoughFilteringTool (){
		configured = false;
	}
	
	public HoughFilteringTool(int radius){
		configured = true;
		radiusOfCircles = radius;
	}
	
	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		HoughFilteringTool filter = new HoughFilteringTool();
		filter.configured = configured;
		filter.radiusOfCircles = radiusOfCircles;
		filter.start = start;
		filter.stop = stop;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Hough Filtering (circle radius = " + radiusOfCircles + ")";
	}

	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		FixedCircleHoughSpace hough = new FixedCircleHoughSpace(1.0, 1.0, imageProcessor.getWidth(), imageProcessor.getHeight(), radiusOfCircles);
		double scale = 1.0 / (stop - start); 
		for (int j = 0;  j < imageProcessor.getHeight(); j++){
			for (int i = 0; i < imageProcessor.getWidth(); i++){
				double value = imageProcessor.getPixelValue(i, j);
				if (value > stop) value = stop;
				value -= start;
				if (value > 0) {
					value *= scale;
					//System.out.println("projecting " + value + " at " + imageProcessor.getPixelValue(i, j));
					hough.fill(i, j, value);
				}
			}
		}
		ImageProcessor imp = hough.getImagePlus().getProcessor();
		return new Grid2D((float[]) imp.getPixels(), imp.getWidth(), imp.getHeight());
	}

	@Override
	public void configure() throws Exception {
		radiusOfCircles = UserUtil.queryInt("Enter radius of circles: ", radiusOfCircles);
		start = UserUtil.queryDouble("Lower Bound of Binarization: ", start);
		stop = UserUtil.queryDouble("Upper bound of binarization: ", stop);
		configured = true;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		return "@inproceedings{Hough59-MAO,\nauthor={Hough P},\ntitle={Machine analysis of bubble chamber pictures},\nbooktitle={International Conference on High Energy Accelerators and Instrumentation},\nyear={1959}\n}";
	}

	@Override
	public String getMedlineCitation() {
		return "Hough P, Machine analysis of bubble chamber pictures. In International Conference on High Energy Accelerators and Instrumentation. 1959";
	}

	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
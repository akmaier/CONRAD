package edu.stanford.rsl.conrad.filtering;

import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.pipeline.IndividualImagePipelineFilteringTool;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * <p> This filter thresholds the values in imageProcessors. 
 * <br>The filter also attempts to replace NaNs pixels via interpolation. 
 * <br>Note. Interpolation is only successful when all adjacent pixels have real values. 
 *
 * @author Rotimi X Ojo 
 */
public class ExtremeValueTruncationFilter extends IndividualImagePipelineFilteringTool {

	private static final long serialVersionUID = -2948507386005078704L;
	private String operation = null;
	private boolean deviceDependent = false;
	public static final String MIN = " min ";
	public static final String MAX = " max ";
	private double threshold = 0;
	
	
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor)
	throws Exception {
		FloatProcessor revan = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight(), imageProcessor.getBuffer());
		revan.setInterpolationMethod(ImageProcessor.BICUBIC);

		for(int y = 0; y < revan.getHeight(); y++){
			for(int x = 0; x < revan.getWidth();x++){
				if(Double.isNaN(revan.getPixelValue(x, y))){
					revan.putPixelValue(x, y, 0);
					revan.putPixelValue(x, y, revan.getInterpolatedPixel(x, y));
				}
				if (operation.equals(MIN) && revan.getPixelValue(x, y)< threshold) {
					revan.putPixelValue(x, y, threshold);
				}
				else if (operation.equals(MAX) && revan.getPixelValue(x, y)>threshold) {
					revan.putPixelValue(x, y, threshold);
				}				
			}
		}
		
		Grid2D out = new Grid2D((float[])revan.getPixels(), revan.getWidth(), revan.getHeight());
		out.setOrigin(imageProcessor.getOrigin());
		out.setSpacing(imageProcessor.getSpacing());
		return out;
	}
	
	@Override
	public IndividualImageFilteringTool clone() {
		ExtremeValueTruncationFilter clone = new ExtremeValueTruncationFilter();
		clone.threshold = threshold;
		clone.operation = operation;
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		if ((operation != null) ) {
			return "Extreme Value Truncation Filter" + operation + " " + threshold;
		} else {
			return "Extreme Value Truncation Filter";
		}
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}
	public String getOperation() {
		return operation;
	}

	public void setOperation(String operation) {
		this.operation = operation;
	}
	
	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}


	@Override
	public void configure() throws Exception {
		String [] operations = {MIN, MAX};
		operation = (String) UserUtil.chooseObject("Select operation: ", "Operation Selection", operations, operation);
		threshold = UserUtil.queryDouble("Enter operand value", threshold);
		if ((operation != null)){
			deviceDependent = UserUtil.queryBoolean("Does this filter model device / hardware dependent behaviour?");
		}
		configured=true;
	}
	
	/**
	 * The use may differ. Hence device dependency can be set. Is set during configuration.
	 */
	@Override
	public boolean isDeviceDependent() {
		return deviceDependent;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

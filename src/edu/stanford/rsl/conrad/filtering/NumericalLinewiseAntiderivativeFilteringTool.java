package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Tool to compute the antiderivative of the projection line-wise.
 * @author akmaier
 *
 */
public class NumericalLinewiseAntiderivativeFilteringTool extends
IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7305539337599312505L;
	private boolean horizontal = true;

	public NumericalLinewiseAntiderivativeFilteringTool(){
		configured = true;
	}

	@Override
	public void configure() throws Exception {
		horizontal = UserUtil.queryBoolean("Compute horizontal?");
		configured =true;
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
	public IndividualImageFilteringTool clone() {
		return new NumericalLinewiseAntiderivativeFilteringTool();
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor)
	throws Exception {
		Grid2D fl = new Grid2D(imageProcessor);
		if (horizontal) {
			for (int j= 0; j < imageProcessor.getHeight(); j++){
				float sum = 0;
				for (int i = 0; i < imageProcessor.getWidth(); i++){
					sum += imageProcessor.getPixelValue(i, j);
					fl.putPixelValue(i, j, sum);
				}
				sum = 0;
				for (int i = 0; i < imageProcessor.getWidth(); i++){
					sum += imageProcessor.getPixelValue(i, j);
					
				}
				sum /= imageProcessor.getWidth();
				for (int i = 0; i < imageProcessor.getWidth(); i++){
					fl.putPixelValue(i, j, fl.getPixelValue(i, j)-sum);
				}
			}
		} else {
			for (int i = 0; i < imageProcessor.getWidth(); i++){
				float sum = 0;
				for (int j= 0; j < imageProcessor.getHeight(); j++){
					sum += imageProcessor.getPixelValue(i, j);
					fl.putPixelValue(i, j, sum);
				}
			}
		}
		return fl;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		return "Antiderivative Tool (line-wise)";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

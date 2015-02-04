package edu.stanford.rsl.conrad.metric;

import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class MeanSquareErrorMetric extends ImageMetric {

	/**
	 * 
	 */
	private static final long serialVersionUID = 668095213557188946L;

	@Override
	public double evaluate() {
		return computeMeanSquareError();
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	protected double computeMeanSquareError(){
		double revan = 0;
		double sum = 0;
		for (int k=0;k<testImage.getStackSize(); k++){
			ImageProcessor rec = testImage.getStack().getProcessor(k+1);
			ImageProcessor ideal = referenceImage.getStack().getProcessor(k+1);
			for (int i=0;i<rec.getWidth();i++){
				for (int j=0;j<rec.getHeight();j++){
					float value = rec.getPixelValue(i, j);
					if (!Float.isInfinite(value) && !Float.isNaN(value)){
						sum++;
						revan += Math.pow(rec.getPixelValue(i, j) - ideal.getPixelValue(i, j),2);
					}
				}
			}
		}
		return revan/sum;
	}

	@Override
	public String toString() {
		return "Mean Square Error";
	}



}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
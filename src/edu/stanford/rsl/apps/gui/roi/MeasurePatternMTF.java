package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;
import ij.gui.Line;
import ij.gui.Plot;

import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Measures the MTF at a regular inlay pattern.
 * 
 * @author akmaier
 *
 */
public class MeasurePatternMTF extends EvaluateROI {
	
	@Override
	public Object evaluate() {
		if (configured) {
			if (roi instanceof Line){
				Line line = (Line) roi;
				double [] pixels = line.getPixels();
				//VisualizationUtil.createPlot("Edge", edge).show();
				double [] fft = FFTUtil.fft(pixels);
				Plot plot = VisualizationUtil.createHalfComplexPowerPlot(fft, "Edge MTF of " + image.getTitle());
				try {
					plot.show();
				} catch (Exception e){
					System.out.println(plot.toString());
				}
			}
		}
		return null;
	}

	public void configure() throws Exception {
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure MTF of a pattern";
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

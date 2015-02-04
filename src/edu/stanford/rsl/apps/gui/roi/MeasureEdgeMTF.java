package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;
import ij.gui.Line;
import ij.gui.Plot;

import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;


public class MeasureEdgeMTF extends EvaluateROI {
	
	double offset = 1;
	
	@Override
	public Object evaluate() {
		if (configured) {
			if (roi instanceof Line){
				Line line = (Line) roi;
				SimpleVector normal = new SimpleVector(line.y2-line.y1,line.x2-line.x1);
				normal.normalizeL2();
				normal.multiplyBy(offset);
				SimpleVector step = normal.clone();
				step.multiplyBy(1/50.0);
				double [] fftAll = null;
				for (int i = 0; i < 100; i++){
					Line newLine = new Line(line.x1-normal.getElement(0)+i*step.getElement(0), line.y1-normal.getElement(1)+i*step.getElement(1), line.x2-normal.getElement(0)+i*step.getElement(0), line.y2-normal.getElement(1)+i*step.getElement(1));
					newLine.setImage(image);
					double [] pixels = newLine.getPixels();
					//VisualizationUtil.createPlot("Line", pixels).show();
					double [] kernel = {-1, 0, 1};
					double [] edge = DoubleArrayUtil.convolve(pixels, kernel);
					//VisualizationUtil.createPlot("Edge", edge).show();
					double [] fft = FFTUtil.fft(edge);
					if (fftAll==null){
						fftAll = fft;
					} else {
						DoubleArrayUtil.add(fftAll, fft);
					}
				}
				DoubleArrayUtil.divide(fftAll, 100);
				Plot plot = VisualizationUtil.createHalfComplexPowerPlot(fftAll, "Edge MTF of " + image.getTitle());
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
		offset = UserUtil.queryDouble("Enter Offset +/- for Averaging in [mm]", offset);
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure MTF of an edge";
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

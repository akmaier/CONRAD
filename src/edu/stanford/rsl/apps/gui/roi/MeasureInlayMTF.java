package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;
import ij.gui.Line;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


public class MeasureInlayMTF extends EvaluateROI {
	
	@Override
	public Object evaluate() {
		if (configured) {
			double centerX = roi.getBounds().getCenterX();
			double centerY = roi.getBounds().getCenterY();
			double radius = roi.getFeretsDiameter() / 2;
			if (roi instanceof Line){
				// Use line as description
				Line line = (Line) roi;
				centerX = line.x1d;
				centerY = line.y1d;
				radius = Math.sqrt(Math.pow(line.x1d - line.x2d, 2) + Math.pow(line.y1d-line.y2d, 2));
			}
			double [] sum = null;
			for (int i =0; i < 360; i ++){ 
				Line line = new Line(centerX, centerY, centerX + Math.cos(Math.PI /180.0 * i) * radius, centerY + Math.sin(Math.PI /180.0 * i) * radius);
				line.setImage(image);
				double [] pixels = line.getPixels();
				double [] kernel = {-1, 0, 1};
				double [] edge = DoubleArrayUtil.convolve(pixels, kernel);
				double [] fft = FFTUtil.fft(edge);
				if (sum == null) {
					sum = fft;
				} else {
					DoubleArrayUtil.add(sum, fft);
				}
				if (debug) System.out.println(i + " " + line.x2d + " " + line.y2d);
			}
			DoubleArrayUtil.divide(sum, 360);
			VisualizationUtil.createHalfComplexPowerPlot(sum, "Inlay MTF of " + image.getTitle()).show();
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
		return "Measure MTF at an phantom inlay";
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
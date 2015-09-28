package edu.stanford.rsl.apps.gui.roi;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;

import ij.ImagePlus;
import ij.process.ImageProcessor;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class LearnHounsfieldScaling extends EvaluateROI {

	private ImagePlus targetImage = null;



	@Override
	public Object evaluate() {
		if (configured){
			System.out.println(roi.getType());
			ImageProcessor mask = roi.getMask();
			Rectangle bounds = roi.getBounds();
			Rectangle bounds2 = targetImage.getRoi().getBounds();
			ArrayList<Point> roiPoints = new ArrayList<Point>();
			if (debug) System.out.println(mask.getWidth() + " " + mask.getHeight() + " " + bounds.x +" " + bounds.y);
			for (int i = 0; i < mask.getWidth(); i++){
				for(int j = 0; j < mask.getHeight(); j++){
					if (mask.getPixelValue(i, j) == 255) {
						Point point = new Point(i,j);
						roiPoints.add(point);
					}
				}
			}
			double [] x = new double[roiPoints.size()];
			double [] y = new double[roiPoints.size()];
			ImageProcessor ip1 = image.getChannelProcessor();
			ImageProcessor ip2 = targetImage.getChannelProcessor();
			for (int i = 0; i < roiPoints.size(); i++){
				Point point = roiPoints.get(i);
				x[i] = ip1.getPixelValue(bounds.x + point.x, bounds.y + point.y);
				y[i] = ip2.getPixelValue(bounds2.x + point.x, bounds2.y + point.y);
			}
			LinearFunction func = new LinearFunction();
			func.fitToPoints(x, y);
			Configuration.getGlobalConfiguration().setHounsfieldScaling(func);
			//double top = func.evaluate(1024);
			//double bottom = func.evaluate(-1024);
			//func.setM(2048.0 / (top - bottom));
			//double t = func.evaluate(0.0);
			System.out.println("Learned: " + func.toString());
			//func.setT(-t);
			VisualizationUtil.createScatterPlot("Hounsfield Mapping", x, y, new LinearFunction()).show();
		}
		return null;
	}

	public void configure() throws Exception {
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		image = (ImagePlus) JOptionPane.showInputDialog(null, "Select image to scale (selection required): ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		targetImage = (ImagePlus) JOptionPane.showInputDialog(null, "Select image with correct scaling: ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		roi = image.getRoi();
		if (roi != null && targetImage.getRoi() != null){
			Rectangle bounds2 = targetImage.getRoi().getBounds();
			//if (roi.getBounds().width == bounds2.width && roi.getBounds().height == bounds2.height) {
				configured = true;
			//}
		}
	}

	@Override
	public String toString() {
		return "Learn Hounsfield Scaling";
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

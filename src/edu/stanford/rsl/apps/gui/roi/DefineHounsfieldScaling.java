package edu.stanford.rsl.apps.gui.roi;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;

import ij.ImagePlus;
import ij.process.ImageProcessor;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients;
import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients.Material;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class DefineHounsfieldScaling extends EvaluateROI {

	private Material material = null;

	@Override
	public Object evaluate() {
		if (configured){
			try {
				System.out.println(roi.getType());
				ImageProcessor mask = roi.getMask();
				Rectangle bounds = roi.getBounds();
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
				ImageProcessor ip1 = image.getChannelProcessor();
				for (int i = 0; i < roiPoints.size(); i++){
					Point point = roiPoints.get(i);
					x[i] = ip1.getPixelValue(bounds.x + point.x, bounds.y + point.y);
				}
				double mean = DoubleArrayUtil.computeMean(x);
				double min = image.getDisplayRangeMin();
				x = new double [] {min, mean};
				double [] y = {-1000, EnergyDependentCoefficients.getCTNumber(material)};
				LinearFunction func = new LinearFunction();
				func.fitToPoints(x, y);
				Configuration.getGlobalConfiguration().setHounsfieldScaling(func);
				System.out.println("Learned: " + func.toString());
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		return null;
	}

	public void configure() throws Exception {
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		image = (ImagePlus) JOptionPane.showInputDialog(null, "Select image with selected material (selection required): ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		material = UserUtil.queryMaterial("Select Material: ", "Material Selection");
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Define Hounsfield Scaling";
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

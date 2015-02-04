package edu.stanford.rsl.apps.gui.roi;

import ij.ImagePlus;
import ij.gui.Plot;
import ij.process.ByteProcessor;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


public class CompareGrayValues extends EvaluateROI {


	private ImagePlus targetImage;

	public void configure() throws Exception {
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		image = (ImagePlus) JOptionPane.showInputDialog(null, "Select image with selection (selection required): ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		targetImage = (ImagePlus) JOptionPane.showInputDialog(null, "Select image copy the selection to: ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public Object evaluate() {
		if (roi.getMask() == null){
			double [] xCoord = new double [roi.getBounds().height*roi.getBounds().width];
			double [] yCoord = new double [roi.getBounds().height*roi.getBounds().width];
			for (int j=0; j < roi.getBounds().height; j++){
				for (int i=0; i < roi.getBounds().width; i++){
					int x = roi.getBounds().x + i;
					int y = roi.getBounds().y + j;
					xCoord[(j*roi.getBounds().width)+i] = image.getProcessor().getPixelValue(x, y);
					yCoord[(j*roi.getBounds().width)+i] = targetImage.getProcessor().getPixelValue(x, y);
				}
			}
			Plot a = VisualizationUtil.createScatterPlot("Compare gray values ", xCoord, yCoord, new LinearFunction());
			a.draw();
			a.show();
		} else {
			// Count pixels in mask
			int count = 0;
			ByteProcessor mask = (ByteProcessor)roi.getMask();
			for (int j=0; j < roi.getBounds().height; j++){
				for (int i=0; i < roi.getBounds().width; i++){
					if (mask.getPixel(i, j) == 255){
						count++;
					}
				}
			}
			double [] xCoord = new double [count];
			double [] yCoord = new double [count];
			int index = 0;
			for (int j=0; j < roi.getBounds().height; j++){
				for (int i=0; i < roi.getBounds().width; i++){
					int x = roi.getBounds().x + i;
					int y = roi.getBounds().y + j;
					if (mask.getPixel(i, j) == 255){
						xCoord[index] = image.getProcessor().getPixelValue(x, y);
						yCoord[index] = targetImage.getProcessor().getPixelValue(x, y);
						index++;
					}
				}
			}
			Plot a = VisualizationUtil.createScatterPlot("Compare gray values ", xCoord, yCoord, new LinearFunction());
			a.draw();
			a.show();
		}
		return null;
	}

	@Override
	public String toString() {
		return "Compare Gray Values";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

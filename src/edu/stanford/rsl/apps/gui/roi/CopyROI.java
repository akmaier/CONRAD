package edu.stanford.rsl.apps.gui.roi;


import ij.ImagePlus;
import ij.gui.Roi;
import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.utils.ImageUtil;


public class CopyROI extends EvaluateROI {

	private ImagePlus targetImage = null;

	@Override
	public Object evaluate() {
		if (configured) {
			Roi copy = (Roi) roi.clone();
			targetImage.setRoi(copy);
		}
		return null;
	}

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
	public String toString() {
		return "Copy ROI to another image";
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

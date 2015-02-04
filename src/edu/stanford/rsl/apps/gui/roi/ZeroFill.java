package edu.stanford.rsl.apps.gui.roi;

import ij.ImagePlus;
import ij.process.ByteProcessor;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * Fills the selected roi with 0.
 * @author akmaier
 *
 */
public class ZeroFill extends EvaluateROI {

	public void configure() throws Exception {
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		image = (ImagePlus) JOptionPane.showInputDialog(null, "Select image with selection (selection required): ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
	}

	@Override
	public Object evaluate() {
		if (roi.getMask() == null){
			for (int j= 0; j < roi.getBounds().height; j++) {
				for (int i= 0; i < roi.getBounds().width; i++){
						image.getProcessor().putPixelValue(roi.getBounds().x + i, roi.getBounds().y+j, 0);
				}
			}
		} else {
			ByteProcessor mask = (ByteProcessor) roi.getMask();
			for (int j= 0; j < mask.getHeight(); j++) {
				for (int i= 0; i < mask.getWidth(); i++){
					if (mask.getPixelValue(i, j) == 255) {
						image.getProcessor().putPixelValue(roi.getBounds().x + i, roi.getBounds().y+j, 0);
					}
				}
			}
		}
		return null;
	}

	@Override
	public String toString() {
		return "Fill with Zeros";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import trainableSegmentation.Weka_Segmentation;
import ij.ImageJ;

/**
 * Class to test the integration of the weka segmentation. Our version of trainable segmentation is rewritten to be able to work with weka 3.6 and does not require Java3D anymore.
 * @author akmaier
 *
 */
public class SegmentationTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		new ImageJ();
		Weka_Segmentation test = new Weka_Segmentation();
		test.run(null);
		
	}

}

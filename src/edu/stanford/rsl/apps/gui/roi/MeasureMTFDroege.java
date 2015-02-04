package edu.stanford.rsl.apps.gui.roi;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import ij.process.ImageStatistics;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * Measures the MTF after Droege et al. "A practical method to measure the MTF of CT scanners". Med Phys 9(5). 758-760. 1982"
 * In order to use the plugin, use the roi manager to define the following ROIs:
 * CT_1 the one material of the MTF measurement
 * CT_2 the other material of the MTF measurement
 * M_1 
 * ... the line pair modules of the phantom
 * M_N
 * @author akmaier
 *
 */
public class MeasureMTFDroege extends EvaluateROI {

	public void configure() throws Exception {
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		image = (ImagePlus) JOptionPane.showInputDialog(null, "Select image to measure MTF: ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		if (RoiManager.getInstance()!=null){
			RoiManager manager = RoiManager.getInstance();
			if (manager.getCount() <= 2){
				IJ.showMessage("Please define at least 3 ROIs");
			} else {
				configured = true;
			}
		} else {
			IJ.showMessage("Please use the ROI Manager to define at least 3 ROIs");
		}
	}

	@Override
	public Object evaluate() {
		RoiManager manager = RoiManager.getInstance();
		Roi [] rois = manager.getRoisAsArray();
		Roi ct1roi = rois[0];
		image.setRoi(ct1roi);
		ImageStatistics stats = image.getStatistics();
		double ct1 = stats.mean;
		double n1 = stats.stdDev;
		Roi ct2roi = rois[1];
		image.setRoi(ct2roi);
		stats = image.getStatistics();
		double ct2 = stats.mean;
		double n2 = stats.stdDev;
		double n0 = Math.sqrt((n1*n1+n2*n2) / 2.0);
		double m0 = Math.abs(ct2-ct1)/2.0;
		for (int i = 2; i < rois.length; i++){
			image.setRoi(rois[i]);
			stats = image.getStatistics();
			double m=Math.sqrt((stats.stdDev*stats.stdDev - n0*n0));
			if (Double.isNaN(m)) m = 0;
			double mtf = Math.PI * Math.sqrt(2) * 0.25 * m / m0;
			System.out.println("Roi " + (i-1) + "\t" + mtf);
		}
		return null;
	}

	@Override
	public String toString() {
		return "Measure Droege MTF";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

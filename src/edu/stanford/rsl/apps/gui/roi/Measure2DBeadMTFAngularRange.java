package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;
import ij.gui.Plot;
import ij.gui.PointRoi;

import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


public class Measure2DBeadMTFAngularRange extends Measure3DBeadMTF {

	double r = 16, start=-15, end=15;
	int sizeOfBead = 10;

	@Override
	public Object evaluate() {
		if (configured) {
			if (roi instanceof PointRoi){
				PointRoi point = (PointRoi) roi;
				double [] sum = null;
				double min = 0;
				double max = 0;
				int px = point.getBounds().x;
				int py = point.getBounds().y;
				int pz = image.getCurrentSlice()-1;
				int smallstep = 1;
				for (int i = (int) start; i < end; i++){
					double beta = ((i*smallstep) / 180.0) * Math.PI;
					// direction vector
					double x = r * Math.cos(beta);
					double y = r * Math.sin(beta);
					double z = 0;
					// Interpolate along line
					double [] pixels = getPixels(image, px, px+x, py,py+y,pz,pz+z, (int) (2*r));
					double [] minmax = DoubleArrayUtil.minAndMaxOfArray(pixels);
					min += minmax[0];
					max += minmax[1];
					double [] mtf = computeMTF(pixels, pixels.length * 16);
					if (sum == null){
						sum = mtf;
					} else {
						DoubleArrayUtil.add(sum, mtf);
					}
				}
				double steps = Math.abs(end - start);
				DoubleArrayUtil.divide(sum, steps);
				min /= steps;
				max /= steps;
				double range = max - min;
				System.out.println("MTF:");
				for (int i=0; i<(sum.length/4); i++) {
					System.out.println(FFTUtil.abs(i, sum));
				}
				double pixelSize = image.getCalibration().pixelWidth;
				double [] measuredFrequencies = computeComplexFrequencies(sum, pixelSize);
				double cutOffFrequency = 1.0 / (2*sizeOfBead*pixelSize);
				int cutOffIndex = 0;
				for (int i = 0; i < measuredFrequencies.length; i++){
					if (measuredFrequencies[i]> cutOffFrequency) {
						cutOffIndex = i;
						break;
					}
				}
				double mtf [] = new double[cutOffIndex];
				double xValues [] = new double[cutOffIndex];
				double modelmtf [] = computeModelMTF(sum, range,sizeOfBead); 
				for(int i=0; i< cutOffIndex;i++){
					mtf[i] = FFTUtil.abs(i, sum)/ FFTUtil.abs(i, modelmtf);
					xValues[i] = measuredFrequencies[i]; 
				}
				Plot plot = VisualizationUtil.createPlot(xValues, mtf, "2D Bead MTF of " + image.getTitle(), "Frequency", "Power");
				try{
					plot.show();
				} catch (Exception e){

				}
			}
		}
		return null;
	}

	@Override
	public void configure() throws Exception {
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			r = UserUtil.queryDouble("Radius in pixels: ", r);
			sizeOfBead = UserUtil.queryInt("Size of Bead in [px]: ", sizeOfBead);
			start = UserUtil.queryDouble("start angle (deg): ", start);
			end = UserUtil.queryDouble("end angle (deg): ", end);
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure 2D Bead MTF in an angular interval";
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

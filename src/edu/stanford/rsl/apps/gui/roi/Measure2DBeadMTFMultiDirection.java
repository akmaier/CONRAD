package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;
import ij.gui.Plot;
import ij.gui.PointRoi;

import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


public class Measure2DBeadMTFMultiDirection extends Measure3DBeadMTF {

	double r = 16, segments = 4;

	@Override
	public Object evaluate() {
		if (configured) {
			if (roi instanceof PointRoi){
				PointRoi point = (PointRoi) roi;
				double [] sum = null;				
				int px = point.getBounds().x;
				int py = point.getBounds().y;
				int pz = image.getCurrentSlice()-1;
				int smallstep = 1;
				for (int j=0; j < segments; j++){
					double intervalSize = 360.0/ (segments*2);
					int start = (int) ((j*intervalSize) - (intervalSize/2));
					int end = (int) ((j*intervalSize) + (intervalSize/2));
					for (int i = (int) start; i < end; i++){
						double beta = ((i*smallstep) / 180.0) * Math.PI;
						// direction vector
						double x = r * Math.cos(beta);
						double y = r * Math.sin(beta);
						double z = 0;
						// Interpolate along line
						double [] pixels = getPixels(image, px, px+x, py,py+y,pz,pz+z, (int) (2*r));
						// remove mean for frequency analysis:
						double [] minmax = DoubleArrayUtil.minAndMaxOfArray(pixels);
						DoubleArrayUtil.add(pixels, -minmax[0]);
						double [] kernel = {-1, 0, 1};
						double [] edge = DoubleArrayUtil.convolve(pixels, kernel);
						// FFT
						double [] fft = FFTUtil.fft(edge);
						if (sum == null){
							sum = fft;
						} else {
							DoubleArrayUtil.add(sum, fft);
						}
					}
					int start2 = (int) (((j+segments)*intervalSize) - (intervalSize/2));
					int end2 = (int) (((j+segments)*intervalSize) + (intervalSize/2));
					for (int i = (int) start; i < end; i++){
						double beta = ((i*smallstep) / 180.0) * Math.PI;
						// direction vector
						double x = r * Math.cos(beta);
						double y = r * Math.sin(beta);
						double z = 0;
						// Interpolate along line
						double [] pixels = getPixels(image, px, px+x, py,py+y,pz,pz+z, (int) (2*r));
						// remove mean for frequency analysis:
						double [] minmax = DoubleArrayUtil.minAndMaxOfArray(pixels);
						DoubleArrayUtil.add(pixels, -minmax[0]);
						double [] kernel = {-1, 0, 1};
						double [] edge = DoubleArrayUtil.convolve(pixels, kernel);
						// FFT
						double [] fft = FFTUtil.fft(edge);
						if (sum == null){
							sum = fft;
						} else {
							DoubleArrayUtil.add(sum, fft);
						}
					}
					DoubleArrayUtil.divide(sum, Math.abs((end - start)+(end2-start2)));
					System.out.println("MTF:");
					for (int i=0; i<(sum.length/4); i++) {
						System.out.println(FFTUtil.abs(i, sum));
					}
					Plot plot = VisualizationUtil.createHalfComplexPowerPlot(sum, "Edge MTF of " + image.getTitle() + " from " + start + " to " + end + " and "+ start2 + " to " + end2 + " (deg)");
					try{
						plot.show();
					} catch (Exception e){

					}
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
			segments = UserUtil.queryDouble("Number of Segments: ", segments);
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure 2D Bead MTF in multiple angular intervals";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

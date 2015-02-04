package edu.stanford.rsl.apps.gui.roi;


import ij.IJ;
import ij.ImagePlus;
import ij.gui.Plot;
import ij.gui.PointRoi;
import ij.process.ImageProcessor;

import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


public class Measure3DBeadMTF extends EvaluateROI {

	private boolean init = false;
	private int vW;
	private int vH;
	private int vD;
	private double r = 16;
	
	public double [] computeComplexFrequencies(double [] fft, double voxelsize){
		double nyquistFrequency = 1 / (2* voxelsize);
		double stepsize = nyquistFrequency / (fft.length/4.0);
		double [] reval = new double [fft.length/4];
		for(int i = 0; i<fft.length/4;i++){
			reval[i]= i * stepsize;
		}
		return reval;
	}
	
	public double [] computeModelMTF(double [] fft, double range, int pixelsize){
		double [] reval = new double [fft.length/2];
		for (int i = 0; i < pixelsize/2; i++)
			reval[i] = range;
		return computeMTF(reval, 0);
	}
	
	public double [] computeMTF(double [] pixels, int padding){
		// remove minimum for frequency analysis:
		double [] minmax = DoubleArrayUtil.minAndMaxOfArray(pixels);
		DoubleArrayUtil.add(pixels, -minmax[0]);
		double [] kernel = {-1, 0, 1};
		double [] edge = DoubleArrayUtil.convolve(pixels, kernel);
		// FFT
		double [] fft = FFTUtil.fft(edge, padding);
		return fft;
	}
	
	
	/**
	 * Method to perform trilinear interpolation along a line through an ImagePlus.
	 * @param image the ImagePlus
	 * @param x1 start x
	 * @param x2 end x
	 * @param y1 start y
	 * @param y2 end y
	 * @param z1 start z
	 * @param z2 end z
	 * @param numberOfQuantizationSteps
	 * @return the array with the interpolated values
	 */
	public double [] getPixels(ImagePlus image, double x1, double x2, double y1, double y2, double z1, double z2, int numberOfQuantizationSteps){
		double [] revan = new double[numberOfQuantizationSteps];
		// direction
		double x = (x2 - x1) / (numberOfQuantizationSteps-1);
		double y = (y2 - y1) / (numberOfQuantizationSteps-1);
		double z = (z2 - z1) / (numberOfQuantizationSteps-1);
		for (int i = 0; i< numberOfQuantizationSteps; i++) {
			revan[i] = trilinear(image, x1 + (i*x), y1+ (i*y), z1+ (i*z));
		}
		return revan;
	}
	
	@Override
	public Object evaluate() {
		if (configured) {
			if (roi instanceof PointRoi){
				PointRoi point = (PointRoi) roi;
				double [] sum = null;				
				int px = point.getBounds().x;
				int py = point.getBounds().y;
				int pz = image.getCurrentSlice()-1;
				int bigstep = 360; // 60;
				int smallstep = 1; //3.0;
				for (int i = 0; i < bigstep; i++){
					double alpha = ((i*smallstep) / 180.0) * Math.PI;
					for (int j = 0; j < bigstep; j++){
						double beta = ((i*smallstep) / 180.0) * Math.PI;
						// direction vector
						double x = r * Math.sin(alpha) * Math.cos(beta);
						double y = r * Math.sin(alpha) * Math.sin(beta);
						double z = r * Math.cos(alpha);
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
				}
				DoubleArrayUtil.divide(sum, Math.pow(bigstep, 2));
				System.out.println("MTF:");
				for (int i=0; i<(sum.length/4); i++) {
					System.out.println(FFTUtil.abs(i, sum));
				}
				Plot plot = VisualizationUtil.createHalfComplexPowerPlot(sum, "Edge MTF of " + image.getTitle());
				try{
				plot.show();
				} catch (Exception e){
					
				}
			} else {
				throw new RuntimeException("A PointRoi is required to measure the 3D MTF.");
			}
		}
		return null;
	}
	
	public void configure() throws Exception {
		
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			r = UserUtil.queryDouble("Radius in pixels: ", r);
			configured = true;
		}
	}

	@Override
	public String toString() {
		return "Measure 3-D MTF of a bead";
	}
	
	/**
	 * Method to initialize the trilinear interpolation.
	 * @param data3D
	 */
	private void init(ImagePlus data3D){
		if (!init){
			vW = data3D.getWidth()-1;
			vH = data3D.getWidth()-1;
			vD = data3D.getStackSize()-1;
			init = true;
		}
	}
	
	/**
	 * Trilinear Interpolation in an ImagePlus.<BR>
	 * Adopted from Volume Viewer by Kai Uwe Barthel: barthel (at) fhtw-berlin.de 
	 * 
	 * This method is initialized in the first call with the volume dimensions to save computation time.<br>
	 * If this interpolation method is used from somewhere else, please use this method only on volumes of the same dimension. Instantiate one Measure3DBeadMTF Object per distinct volume dimension.
	 * 
	 * @param data3D the ImagePlus
	 * @param x the x coordinate
	 * @param y the y coordinate
	 * @param z the z coordinate
	 * @return the interpolated value
	 */
	public double trilinear(ImagePlus data3D,  double x, double y, double z) {
		
		init(data3D);
		
		int tx = (int)x;
		double dx = x - tx;
		int tx1 = (tx < vW) ? tx+1 : tx;
		int ty = (int)y;
		double dy = y - ty;
		int ty1 = (ty < vH) ? ty+1 : ty;
		int tz = (int)z;
		double dz = z - tz;
		int tz1 = (tz < vD) ? tz+1 : tz;
		
		ImageProcessor ptz = data3D.getStack().getProcessor(tz+1);
		ImageProcessor ptz1 = data3D.getStack().getProcessor(tz1+1);
		
		float  v000 = ptz.getPixelValue(tx, ty);
		float  v001 = ptz1.getPixelValue(tx, ty); 
		float  v010 = ptz.getPixelValue(tx, ty1); 
		float  v011 = ptz1.getPixelValue(tx, ty1); 
		float  v100 = ptz.getPixelValue(tx1, ty); 
		float  v101 = ptz1.getPixelValue(tx1, ty); 
		float  v110 = ptz.getPixelValue(tx1, ty1); 
		float  v111 = ptz1.getPixelValue(tx1, ty1); 
		
		return (
				(v100 - v000)*dx + 
				(v010 - v000)*dy + 
				(v001 - v000)*dz +
				(v110 - v100 - v010 + v000)*dx*dy +
				(v011 - v010 - v001 + v000)*dy*dz +
				(v101 - v100 - v001 + v000)*dx*dz +
				(v111 + v100 + v010 + v001 - v110 - v101 - v011 - v000)*dx*dy*dz + v000 );
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
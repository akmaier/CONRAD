package edu.stanford.rsl.conrad.utils;

import ij.process.FHT;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;


/**
 * This class is currently unused as all transforms in OSCAR are now based on JTransforms
 * However some of these methods may come in handy when dealing with the ImageJ implementation of
 * the FHT.
 * 
 * Many of the methods are very similar to the ones in ij/plugin/FFT.java. If code was partially
 * taken from there this is declared in the comment before the method.
 * 
 * @author Andreas Maier
 *
 */
public abstract class FHTUtil {
	/**
	 * Image is padded with 0. Code was partially taken from ij.plugin.FFT.java::pad(). 
	 * 
	 * Thanks for the inspiration!
	 * 
	 * @param ip the ImageProcessor to be padded.
	 * @return the padded ImageProcessor.
	 */
	public static ImageProcessor padImageProcessor(ImageProcessor ip, int maxN) {
		ImageProcessor ip2 = ip.createProcessor(maxN, maxN);
		ip2.setValue(0);
		ip2.fill();
		ip2.insert(ip, 0, 0);
		return ip2;
	}

	/**
	 * Image is padded with 0. Size of padding is automatically determined. This means a slight increase in computation time. 
	 * 
	 * Code was partially taken from ij.plugin.FFT.java::pad().
	 * Thanks for the inspiration!
	 * 
	 * @param ip the ImageProcessor to be padded.
	 * @return the padded ImageProcessor.
	 * 
	 */
	public static ImageProcessor padImageProcessor(ImageProcessor ip) {
		int maxN = FFTUtil.getNextPowerOfTwo(Math.max(ip.getWidth(), ip.getHeight()));
		ImageProcessor ip2 = ip.createProcessor(maxN, maxN);
		ip2.setValue(0);
		ip2.fill();
		ip2.insert(ip, 0, 0);
		return ip2;
	}

	/**
	 * Converts real and imaginary parts of an FFT to Hartley domain.
	 * @param real FloatProcessor with the real values
	 * @param imag FloatProcessor with the imaginary values
	 * @return the values in Hartley Domain
	 * 
	 */
	public static FHT complexToFHT(FloatProcessor real, FloatProcessor imag){
		FHT fht = new FHT(new FloatProcessor(real.getWidth(),real.getHeight()));
		float [] fhtPixels = (float[]) fht.getPixels();
		float [] imagPixels = (float[]) imag.getPixels();
		float [] realPixels = (float[]) real.getPixels();
		for(int j=0; j<fht.getHeight();j++){
			cplxFHT(j,fht.getWidth(), realPixels, imagPixels, false, fhtPixels);
		}
		fht.swapQuadrants();
		return fht;
	}

	/** Build FHT input for equivalent FFT input
	 *   @author Joachim Wesner
	 */
	private static void cplxFHT(int row, int maxN, float[] re, float[] im, boolean reim, float[] fht) {
		int base = row*maxN;
		int offs = ((maxN-row)%maxN) * maxN;
		if (!reim) {
			for (int c=0; c<maxN; c++) {
				int l =  offs + (maxN-c)%maxN;
				fht[base+c] = ((re[base+c]+re[l]) - (im[base+c]-im[l]))*0.5f;
			}
		} else {
			for (int c=0; c<maxN; c++) {
				int l = offs + (maxN-c)%maxN;
				fht[base+c] = ((im[base+c]+im[l]) + (re[base+c]-re[l]))*0.5f;
			}
		}
	}

	/**	 FFT real value of one row from 2D Hartley Transform.
	 *	@author Joachim Wesner
	 */
	private static void FHTreal(int row, int maxN, float[] fht, float[] real) {
		int base = row*maxN;
		int offs = ((maxN-row)%maxN) * maxN;
		for (int c=0; c<maxN; c++) {
			real[base+c] = (fht[base+c] + fht[offs+((maxN-c)%maxN)])*0.5f;
		}
	}


	/** FFT imag value of one row from 2D Hartley Transform.
	 *	@author Joachim Wesner
	 */
	private static void FHTimag(int row, int maxN, float[] fht, float[] imag) {
		int base = row*maxN;
		int offs = ((maxN-row)%maxN) * maxN;
		for (int c=0; c<maxN; c++) {
			imag[base+c] = (-fht[base+c] + fht[offs+((maxN-c)%maxN)])*0.5f;
		}
	}

	/**
	 * Computes the imaginary values of an FFT given the values of an FHT.
	 * 
	 * @param fht the FHT as input
	 * @return the imaginary values as FloatProcessor. Use .log() if you want an improved visualization of the result.
	 */
	public static FloatProcessor imaginaryFromFHT(FHT fht){
		FHT img = new FHT(new FloatProcessor(fht.getWidth(),fht.getHeight()));
		float [] fhtPixels = (float[]) fht.getPixels();
		float [] imagPixels = (float[]) img.getPixels();
		for(int j=0; j<fht.getHeight();j++){
			FHTimag(j,fht.getWidth(), fhtPixels, imagPixels);
		}
		img.swapQuadrants();
		return img;
	}

	/**
	 * Computes the real values of an FFT given the values of an FHT.
	 * 
	 * @param fht the FHT as input
	 * @return the real values as FloatProcessor. Use .log() if you want an improved visualization of the result.
	 * 
	 */
	public static FloatProcessor realFromFHT(FHT fht){
		FHT img = new FHT(new FloatProcessor(fht.getWidth(),fht.getHeight()));
		float [] fhtPixels = (float[]) fht.getPixels();
		float [] realPixels = (float[]) img.getPixels();
		for(int j=0; j<fht.getHeight();j++){
			FHTreal(j,fht.getWidth(), fhtPixels, realPixels);
		}
		img.swapQuadrants();
		return img;
	}

	/**
	 * Computes the power spectrum of an FFT given the values of an FHT.
	 * 
	 * @param fht the FHT as input
	 * @return the power spectrum as FloatProcessor. Use .log() if you want an improved visualization of the result.
	 * 
	 */
	public static ImageProcessor powerFromFHT(FHT fht){
		ImageProcessor imag = imaginaryFromFHT(fht);
		ImageProcessor real = realFromFHT(fht);
		ImageProcessor power = new FloatProcessor(fht.getWidth(),fht.getHeight());
		float [] realPixels = (float[]) real.getPixels();
		float [] imagPixels = (float[]) imag.getPixels();
		float [] powerPixels = (float []) power.getPixels();
		for(int j=0; j<(fht.getHeight()*fht.getWidth());j++){
			double value = Math.sqrt(Math.pow(imagPixels[j], 2) + Math.pow(realPixels[j], 2));
			powerPixels[j] = (float) value;
		}
		return power;
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
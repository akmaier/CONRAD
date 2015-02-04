

import java.awt.Frame;
import java.util.ArrayList;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Plot;
import ij.gui.PointRoi;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

import javax.swing.JOptionPane;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;


public class Measure_MTF_Wire implements PlugIn {

	public static ArrayList<ImagePlus> getAvailableImagePlus(){
		ArrayList<ImagePlus> list = new ArrayList<ImagePlus>();
		Frame [] frames = ImageJ.getFrames();
		for (Frame frame: frames){
			if (frame instanceof ImageWindow){
				ImageWindow window = (ImageWindow)frame;
				if (! window.isClosed()){
					list.add(window.getImagePlus());
				}
			}
		}
		return list;
	}

	public static ImagePlus [] getAvailableImagePlusAsArray(){
		ArrayList<ImagePlus> list = getAvailableImagePlus();
		ImagePlus [] array = new ImagePlus[list.size()];
		for(int i=0; i< list.size(); i++){
			array[i] = list.get(i);
		}
		return array;
	}


	@Override
	public String toString() {
		return "Measure Droege MTF";
	}

	@Override
	public void run(String arg) {
		try {
			configure();
			evaluate();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	double r = 16, start=-15, end=15;
	int sizeOfBead = 10;
	boolean configured = true;
	ImagePlus image;
	Roi roi;
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
	private boolean init = false;
	private int vW;
	private int vH;
	private int vD;

	
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
	/**
	 * Returns the minimal and the maximal value in a given array
	 * @param array the array
	 * @return an array with minimal and maximal value
	 */
	public static double [] minAndMaxOfArray(double [] array){
		double [] revan = new double [2];
		revan[0] = Double.MAX_VALUE;
		revan[1] = -Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++){
			if (array[i] < revan[0]) {
				revan[0] = array[i];
			}
			if (array[i] > revan[1]) {
				revan[1] = array[i];
			}
		}
		return revan;		
	}
	
	/**
	 * Adds a constant to the first array
	 * @param sum the first array
	 * @param toAdd the constant to add
	 */
	public static double [] add(double[] sum, double toAdd) {
		for (int i =0; i < sum.length; i++){
			sum[i] += toAdd;
		}
		return sum;
	}
	
	/**
	 * Performs a 1-D convolution of the input array with the kernel array.<BR>
	 * New array will be only of size <br>
	 * <pre>
	 * output.lenght = input.length - (2 * (kernel.length/2));
	 * </pre>
	 * (Note that integer arithmetic is used here)<br>
	 * @param input the array to be convolved
	 * @param kernel the kernel
	 * @return the output array.
	 */
	public static double [] convolve(double [] input, double [] kernel){
		int offset = ((kernel.length) / 2);
		double [] revan = new double [input.length - (2* offset)];
		double weightSum = 0;
		for (int j = 0; j < kernel.length; j++){
			weightSum += kernel[j];
		}
		if (weightSum == 0) weightSum = 1; 
		for (int i = offset; i < input.length-offset;i++){
			double sum = 0;
			for (int j = -offset; j <= offset; j++){
				sum += kernel[offset+j] * input[i+j];
			}
			sum /= weightSum;
			revan [i-offset] = sum;
		}
		return revan;
	}
	/**
	 * Returns the next power of 2 given a certain integer value 
	 * 
	 * Code was partially taken from ij.plugin.FFT.java::pad().
	 * Thanks for the inspiration!
	 * 
	 * @param value the input number.
	 * @return the next power of two.
	 */
	public static int getNextPowerOfTwo(int value){
		int i = 2;
		while (i < value) {
			i *= 2;
		}
		return i;
	}

	
	public static double []  fft(double []  array, int padding){
		DoubleFFT_1D fft = new DoubleFFT_1D(getNextPowerOfTwo(array.length + padding));
		double [] test = new double [getNextPowerOfTwo(array.length+padding) * 2];
		for (int i = 0; i < array.length; i++){
			test[i*2] = array[i];
		}
		fft.complexForward(test);
		return test;
	}
	
	public double [] computeMTF(double [] pixels, int padding){
		// remove minimum for frequency analysis:
		double [] minmax = minAndMaxOfArray(pixels);
		add(pixels, -minmax[0]);
		double [] kernel = {-1, 0, 1};
		double [] edge = convolve(pixels, kernel);
		// FFT
		double [] fft = fft(edge, padding);
		return fft;
	}
	
	/**
	 * Adds one array to the first array
	 * @param sum the first array
	 * @param toAdd the array to add
	 */
	public static void add(double[] sum, double[] toAdd) {
		for (int i =0; i < sum.length; i++){
			sum[i] += toAdd[i];
		}
	}
	
	/**
	 * Divides all entries of array by divident.
	 * @param array the array
	 * @param divident the number used for division.
	 */
	public static double [] divide(double[] array, double divident) {
		for (int i =0; i < array.length; i++){
			array[i] /= divident;
		}
		return array;
	}
	
	
	public double [] computeComplexFrequencies(double [] fft, double voxelsize){
		double nyquistFrequency = 1 / (2* voxelsize);
		double stepsize = nyquistFrequency / (fft.length/4.0);
		double [] reval = new double [fft.length/4];
		for(int i = 0; i<fft.length/4;i++){
			reval[i]= i * stepsize;
		}
		return reval;
	}
	
	/**
	 * Computes the absolute value of the complex number at position pos in the array
	 * @param pos the position
	 * @param array the array which contains the values
	 * @return the absolute value
	 */
	public static double abs (int pos, double[] array){
		return Math.sqrt(Math.pow(array[pos *2], 2) + Math.pow(array[(2*pos)+1], 2));
	}
	
	public double [] computeModelMTF(double [] fft, double range, int pixelsize){
		double [] reval = new double [fft.length/2];
		for (int i = 0; i < pixelsize/2; i++)
			reval[i] = range;
		return computeMTF(reval, 0);
	}
	
	public static Plot createPlot(double [] xValues, double [] yValues, String title, String xLabel, String yLabel){
		double miny = Double.MAX_VALUE;
		double maxy = -Double.MAX_VALUE;
		double minx = Double.MAX_VALUE;
		double maxx = -Double.MAX_VALUE;
		for (int i = 0; i < xValues.length; i ++){
			miny = (yValues[i] < miny) ? yValues[i] : miny;
			maxy = (yValues[i] > maxy) ? yValues[i] : maxy;
			minx = (xValues[i] < minx) ? xValues[i] : minx;
			maxx = (xValues[i] > maxx) ? xValues[i] : maxx;

		}
		if (miny == maxy){
			maxy++;
		}
		if (minx == maxx){
			maxx++;
		}
		Plot plot = new Plot(title, xLabel, yLabel, xValues, yValues, Plot.DEFAULT_FLAGS);
		plot.setLimits(minx, maxx, miny, maxy);
		return plot;
	}
	
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
					double [] minmax = minAndMaxOfArray(pixels);
					min += minmax[0];
					max += minmax[1];
					double [] mtf = computeMTF(pixels, pixels.length * 16);
					if (sum == null){
						sum = mtf;
					} else {
						add(sum, mtf);
					}
				}
				double steps = Math.abs(end - start);
				divide(sum, steps);
				min /= steps;
				max /= steps;
				double range = max - min;
				System.out.println("MTF:");
				for (int i=0; i<(sum.length/4); i++) {
					System.out.println(abs(i, sum));
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
					mtf[i] = abs(i, sum)/ abs(i, modelmtf);
					xValues[i] = measuredFrequencies[i]; 
				}
				Plot plot = createPlot(xValues, mtf, "2D Bead MTF of " + image.getTitle(), "Frequency", "Power");
				try{
					plot.show();
				} catch (Exception e){

				}
			}
		}
		return null;
	}

	/**
	 * Queries the User for an Integer value using Swing.
	 * @param message
	 * @param initialValue
	 * @return the chosen int
	 * @throws Exception
	 */
	public static int queryInt(String message, int initialValue) throws Exception{
		String input = JOptionPane.showInputDialog(message, "" + initialValue);
		if (input == null) throw new Exception("Selection aborted");
		return Integer.parseInt(input);
	}
	
	/**
	 * Queries the User for a Double values using Swing.
	 * @param message
	 * @param initialValue
	 * @return the chosen double
	 * @throws Exception
	 */
	public static double queryDouble(String message, double initialValue) throws Exception{
		String input = JOptionPane.showInputDialog(message, "" + initialValue);
		if (input == null) throw new Exception("Selection aborted");
		return Double.parseDouble(input);
	}

	public void configure() throws Exception {
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			r = queryDouble("Radius in pixels: ", r);
			sizeOfBead = queryInt("Size of Bead in [px]: ", sizeOfBead);
			start = queryDouble("start angle (deg): ", start);
			end = queryDouble("end angle (deg): ", end);
			configured = true;
		}
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
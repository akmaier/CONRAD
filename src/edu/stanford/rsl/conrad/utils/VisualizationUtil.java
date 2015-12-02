package edu.stanford.rsl.conrad.utils;

import java.awt.Color;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.fitting.Function;
import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Line;
import ij.gui.Overlay;
import ij.gui.Plot;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public abstract class VisualizationUtil {
	
	public static synchronized Plot createPlot(String title, double [] yValues){
		return createPlot(yValues, title, "Column", "Power");
	}
	
	public static synchronized Plot createPlot(String title, float [] yValues){
		double[] tmp = new double[yValues.length];
		for(int i=0; i<yValues.length; ++i)
			tmp[i] = yValues[i];
		return createPlot(tmp, title, "Column", "Power");
	}
	
	public static synchronized Plot createPlot(float [] yValues){
		double[] tmp = new double[yValues.length];
		for(int i=0; i<yValues.length; ++i)
			tmp[i] = yValues[i];
		return createPlot(tmp, "Current Plot", "Column", "Power");
	}
	
	public static synchronized Plot createPlot(double [] yValues){
		return createPlot(yValues, "Average Row Weighting", "Column", "Power");
	}
	
	public static Plot createSplinePlot(BSpline spline){
		int length = 100;
		double [] x = new double[length];
		double [] y = new double[length];
		for (int i = 0; i< length; i++){
			PointND p = spline.evaluate(((double) i) / (length));
			x[i] = p.get(0);
			y[i] = p.get(1);
		}
		
		return new Plot("Spline Plot", "x", "y", x, y);
	}
	
	public static Plot createPlot(double [] yValues, String title, String xLabel, String yLabel){
		double [] xValues = new double [yValues.length];
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < xValues.length; i ++){
			min = (yValues[i] < min) ? yValues[i] : min;
			max = (yValues[i] > max) ? yValues[i] : max;
			xValues[i] = i + 1;
		}
		if (min == max){
			max++;
		}
		Plot plot = new Plot(title, xLabel, yLabel, xValues, yValues, Plot.DEFAULT_FLAGS);
		plot.setLimits(1, xValues.length, min, max);
		return plot;
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
	
	public static Plot createComplexPowerPlot(double [] yValues, String title){
		double [] absValues = new double [yValues.length / 2];
		for (int i = 0; i < absValues.length; i ++){
			absValues[i] = FFTUtil.abs(i, yValues);
		}
		return createPlot(absValues, title, "Frequency (Center = +/- Nyquist)", "Power");
	}
	
	public static Plot createHalfComplexPowerPlot(double [] yValues, String title){
		double [] absValues = new double [yValues.length / 4];
		for (int i = 0; i < absValues.length; i ++){
			absValues[i] = FFTUtil.abs(i, yValues);
		}
		return createPlot(absValues, title, "Frequency", "Power");
	}
	
	
	public static Plot createHalfComplexPowerPlot(double [] yValues, double [] xValues, String title){
		double [] absValues = new double [yValues.length / 4];
		for (int i = 0; i < absValues.length; i ++){
			absValues[i] = FFTUtil.abs(i, yValues);
		}
		return createPlot(xValues, absValues, title, "Frequency", "Power");
	}
	
	public static Plot createComplexPowerPlot(double [] yValues){
		return createComplexPowerPlot(yValues, "Complex Power Plot");
	}

	public static ImagePlus showGrid2D(Grid2D grid, String title){
		if (grid instanceof MultiChannelGrid2D){
			MultiChannelGrid2D multiChannelGrid2D = (MultiChannelGrid2D) grid;
			ImagePlus imp = new ImagePlus();
			ImageStack stack = new ImageStack(grid.getWidth(),grid.getHeight());
			for (int c = 0; c < multiChannelGrid2D.getNumberOfChannels(); c++){
				stack.addSlice(multiChannelGrid2D.getChannelNames()[c],ImageUtil.wrapGrid2D(multiChannelGrid2D.getChannel(c)));
			}
			imp.setStack(title, stack);
			imp.show();	
			return imp;
		}
		return showImageProcessor(ImageUtil.wrapGrid2D(grid), title);
	}
	
	public static ImagePlus showImageProcessor(ImageProcessor image, String title){
		ImagePlus imp = new ImagePlus();
		ImageStack stack = new ImageStack(image.getWidth(),image.getHeight());
		stack.addSlice(title, image);
		imp.setStack(title, stack);
		imp.show();
		return imp;
	}
	
	public static ImagePlus showImageProcessor(ImageProcessor image){
		return showImageProcessor(image, "Untitled Image");
	}
	
	public static ImagePlus showGrid3DX( Grid3D image, String title){
		
		int[] size = image.getSize(); 
		int nSlices = size[0];
		int width = size[1];
		int height = size[2];
				
		System.out.print("Grid size: " + width + " X " + height + " X " +  nSlices +  ".\n" );
		
		//this is more memory efficient to use grid3D directly
		new ImageJ();
		ImagePlus imp = ImageUtil.wrapGrid3D(image, "");
		imp.show();
		imp.setTitle( title );
		
		/*
		ImagePlus imp = new ImagePlus();
		ImageStack stack = new ImageStack( width, height);
		
		for( int n = 0 ; n < nSlices; n++){
			//System.out.print("Slice " + n + " \n" );
			FloatProcessor slice = new FloatProcessor( width, height);
			for (int y=0; y<height; y++) {
				for (int x=0; x<width; x++) {
					slice.setf(x, y, image.getAtIndex(n, x, y));
				}
			}
						
			 stack.addSlice( " " + n, slice );
		}
		imp.setStack(title, stack);
		imp.show();
		*/
		return imp;
	}
	
	public static ImagePlus showGrid3DX( Grid3D image){
		return showGrid3DX( image,  "Untitled Image");
	}
	
	public static ImagePlus showGrid3DZ( Grid3D image, String title){
		
		int[] size = image.getSize(); 
		int width = size[0];
		int height = size[1];
		int nSlices = size[2];
				
		System.out.print("Grid size: " + width + " X " + height + " X " +  nSlices +  ".\n" );
		
		//not sure if imageJ could rotate the grid3D data, so i copy and rearrange the 3D image to show cornal view
		
		new ImageJ();
		ImagePlus imp = new ImagePlus();
		ImageStack stack = new ImageStack( width, height);
		
		for( int n = 0 ; n < nSlices; n++){
			//System.out.print("Slice " + n + " \n" );
			FloatProcessor slice = new FloatProcessor( width, height);
			for (int y=0; y<height; y++) {
				for (int x=0; x<width; x++) {
					slice.setf(x, y, image.getAtIndex(x, y, n));
				}
			}
						
			 stack.addSlice( " " + n, slice );
		}
		imp.setStack(title, stack);
		imp.show();

		return imp;
	}
	
	public static ImagePlus showGrid3DZ( Grid3D image){
		return showGrid3DZ( image,  "Untitled Image");
	}
	
	/**
	 * Takes an image overlay for Image Plus images and draws a cross at the 2D position defined by pos.
	 * @param ov the overlay to draw on
	 * @param pos the position where the cross is to be drawn
	 * @param slice the slice number where the point should be drawn
	 * @param crossSize half the width of the cross' bounding box 
	 * @param col the color of the cross
	 */
	public static void printCrossAtPoint(Overlay ov, PointND pos, int slice, double crossSize, Color col){
		double pos0 = pos.get(0)+0.5;
		double pos1 = pos.get(1)+0.5;
		Line line = new Line(pos0, pos1-crossSize, pos0, pos1+crossSize);
		line.setStrokeWidth(1);
		line.setStrokeColor(col);
		line.setPosition(slice+1);
		line.setStrokeWidth(0.5);
		ov.add(line);

		line = new Line(pos0-crossSize, pos1, pos0+crossSize, pos1);
		line.setStrokeWidth(1);
		line.setStrokeColor(col);
		line.setPosition(slice+1);
		line.setStrokeWidth(0.5);
		ov.add(line);
	}
	

	public static Plot plotCompareGrayValues(ImageProcessor before, ImageProcessor after, Function func){
		double [] xp = new double[before.getWidth()*before.getHeight()];
		double [] yp = new double[before.getWidth()*before.getHeight()];
		for (int i=0;i<before.getWidth();i++){
			for (int j=0;j<before.getHeight();j++){
				xp[before.getHeight()*i+j] = before.getPixelValue(i, j);
				yp[before.getHeight()*i+j] = after.getPixelValue(i, j);
			}
		}
		return createScatterPlot(xp, yp, func);
	}
	
	public static Plot createScatterPlot(double [] xCoords, double [] yCoords, Function func){
		return createScatterPlot("Plot", xCoords, yCoords, func);
	}
	
	public static Plot createScatterPlot(String title, double [] xCoords, double [] yCoords, Function func){
		double margin = 0.15;
		int scale = 200;
		func.fitToPoints(xCoords, yCoords);
		float [] xVals = new float[scale];
		float [] yVals = new float[scale];
		double [] stats = DoubleArrayUtil.minAndMaxOfArray(xCoords);
		double range = (stats[1] - stats[0]) * (1+(2*margin));
		for (int i=0;i<scale;i++){
			double x = stats[0] - (range * margin) + ((i + 0.0) / scale) * range;
			xVals[i] = (float) x;
			yVals[i] = (float) func.evaluate(x);
		}
		double avg = 0;
		for (int i=0;i<xCoords.length;i++){
			double other = func.evaluate(xCoords[i]);
			avg += Math.pow(other - yCoords[i],2);
		}
		avg /= xCoords.length;
		avg = Math.sqrt(avg);
		
		Plot plot = new Plot(title + " (r = "+DoubleArrayUtil.correlateDoubleArrays(xCoords, yCoords)+" SSIM: "  + DoubleArrayUtil.computeSSIMDoubleArrays(xCoords, yCoords)  + ") Model: " + func.toString() + " standard deviation along model: " + avg, "X", "Y", xVals, yVals, Plot.DEFAULT_FLAGS);		
		plot.addPoints(xCoords, yCoords, Plot.CIRCLE);
		return plot;
	}
	
	/**
	 * Creates a scatter plot for two given double arrays with the given title.
	 * @param title The Title
	 * @param xCoords the array for the x-coordinates
	 * @param yCoords the array for the y-coordinates
	 * @return the resulting plot.
	 */
	public static Plot createScatterPlot(String title, double [] xCoords, double [] yCoords){
		Function func = new LinearFunction();
		func.fitToPoints(xCoords, yCoords);
		double margin = 0.15;
		int scale = 200;
		float [] xVals = new float[scale];
		float [] yVals = new float[scale];
		double [] stats = DoubleArrayUtil.minAndMaxOfArray(xCoords);
		double range = (stats[1] - stats[0]) * (1+(2*margin));
		for (int i=0;i<scale;i++){
			double x = stats[0] - (range * margin) + ((i + 0.0) / scale) * range;
			xVals[i] = (float) x;
			yVals[i] = (float) func.evaluate(x);
		}
		Plot plot = new Plot(title + "Plot (r = "+DoubleArrayUtil.correlateDoubleArrays(xCoords, yCoords)+")", "X", "Y", xVals, yVals, Plot.DEFAULT_FLAGS);
		plot.addPoints(xCoords, yCoords, Plot.CROSS);
		return plot;
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
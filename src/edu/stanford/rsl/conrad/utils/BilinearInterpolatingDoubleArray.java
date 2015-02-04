package edu.stanford.rsl.conrad.utils;

import java.io.Serializable;

import ij.ImageJ;
import ij.process.FloatProcessor;


/**
 * Class for interpolation on an arbitrary regular 2-D grid.
 * (Inplementation follows formula as displayed in Wikipedia.)
 * 
 * @see <a href="http://en.wikipedia.org/wiki/Bilinear_interpolation">http://en.wikipedia.org/wiki/Bilinear_interpolation</a>
 * 
 * @author akmaier
 *
 */
public class BilinearInterpolatingDoubleArray implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1183806565056190746L;
	private double [] meshx;
	private double [] meshy;
	private double [][] values;
	private double minx;
	private double maxx;
	private double miny;
	private double maxy;
	
	/**
	 * Constructor with requires an x-mesh, a y-mesh, and a two-dimensional array of values at the mesh points
	 * @param meshx the mesh spacing in x direction
	 * @param meshy the mesh spacing in y direction
	 * @param values the values at the mesh points.
	 */
	public BilinearInterpolatingDoubleArray(double [] meshx, double [] meshy, double [][] values){
		this.meshx = meshx;
		this.meshy = meshy;
		this.values = values;
		double [] minmax = DoubleArrayUtil.minAndMaxOfArray(meshx);
		minx = minmax[0];
		maxx = minmax[1];
		minmax = DoubleArrayUtil.minAndMaxOfArray(meshy);
		miny = minmax[0];
		maxy = minmax[1];
	}
	
	/**
	 * Interpolate an arbitrary point between the meshes.
	 * 
	 * @param meshPointX the x-coordinate
	 * @param meshPointY the y-coordinate
	 * @return the interpolated value
	 * @throws Exception may happen if the point is not within the grid.
	 */
	public double getValue(double meshPointX, double meshPointY) throws Exception{
		if ((meshPointX < minx)||(meshPointX > maxx)){
			throw new Exception("Cannot extrapolate outside mesh x-axis: " + meshPointX + " Range: [ " + minx + ", " + maxx + " ]");
		}
		if ((meshPointX < miny)||(meshPointY > maxy)){
			throw new Exception("Cannot extrapolate outside mesh y-axis: " + meshPointY + " Range: [ " + miny + ", " + maxy + " ]");
		}
		int lowerIndexx = 0;
		for (int i = 1; i < meshx.length; i++){
			if (meshx[i] > meshPointX){
				lowerIndexx = i - 1;
				break;
			}
		}
		int lowerIndexy = 0;
		for (int i = 1; i < meshy.length; i++){
			if (meshy[i] > meshPointY){
				lowerIndexy = i - 1;
				break;
			}
		}
		// denominator in interpolation formula (area of the interpolation patch)
		double denominator = (meshx[lowerIndexx+1] - meshx[lowerIndexx])*(meshy[lowerIndexy+1] - meshy[lowerIndexy]);
		// Interpolation value at values[lowerIndexx][lowerIndexy]
		double revan = values[lowerIndexx][lowerIndexy] * (meshx[lowerIndexx+1] - meshPointX) * (meshy[lowerIndexy+1] - meshPointY);
		// Interpolation value at values[lowerIndexx+1][lowerIndexy]
		revan += values[lowerIndexx+1][lowerIndexy] * (meshPointX - meshx[lowerIndexx]) * (meshy[lowerIndexy+1] - meshPointY);
		// Interpolation value at values[lowerIndexx][lowerIndexy+1]
		revan += values[lowerIndexx][lowerIndexy+1] * (meshx[lowerIndexx+1] - meshPointX) * (meshPointY - meshy[lowerIndexy]);
		// Interpolation value at values[lowerIndexx+1][lowerIndexy+1]
		revan += values[lowerIndexx+1][lowerIndexy+1] * (meshPointX - meshx[lowerIndexx]) * (meshPointY - meshy[lowerIndexy]);
		// devide by area;
		revan /= denominator;
		return revan;
	}
	
	/**
	 * Code for testing using CONRAD Software package. Will create the example image as displayed in Wikipedia.<BR>
	 * (http://en.wikipedia.org/wiki/File:Bilininterp.png)
	 * 
	 * @param args
	 */
	public static void main (String[] args){
		double [] array1 =  {0 ,1};
		double [] array2 = {0, 1};
		double [] [] arrayVal = { {1, 0}, {0.5, 1}};
		BilinearInterpolatingDoubleArray bil = new BilinearInterpolatingDoubleArray(array1, array2, arrayVal);
		FloatProcessor fl = bil.toFloatProcessor(640, 480);
		VisualizationUtil.showImageProcessor(fl,"Interpolation Example as in Wikipedia");
	}
	
	/**
	 * Renders a float processor of the 2-D array given the number of bins in x and y direction
	 * @param binsX the number of bins in x direction
	 * @param binsY the number of bins in y direction
	 * @return the rendered FloatProcessor.
	 */
	public FloatProcessor toFloatProcessor(int binsX, int binsY){
		float [] pixels = new float [binsX * binsY];
		double rangeX = maxx - minx;
		double rangeY = maxy - miny;
		for (int i= 0; i < binsX; i++){
			for (int j = 0; j < binsY; j++){
				try {
					double coordX = minx + ((((double)i)/binsX) * rangeX);
					double coordY = miny + ((((double)j)/binsY) * rangeY);
					pixels[(j*binsX)+i] = (float) this.getValue(coordX, coordY);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		new ImageJ();
		return new FloatProcessor(binsX, binsY, pixels, ImageUtil.getDefaultColorModel());
	}
	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
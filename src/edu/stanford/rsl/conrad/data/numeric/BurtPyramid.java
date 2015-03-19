/*
 * Copyright (C) 2010-2014 - Mathias Unberath & Eric Goppert
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.filter.GaussianBlur;
import ij.process.ImageProcessor;

public class BurtPyramid{
	
	private ArrayList<Grid2D[]> precomp = null;
	private boolean PRECOMPUTED = false;
	
	private ImagePlus level0 = null;
	private int nOctaves = 0;
	private int lvlPerOctave = 1;
	
	//===============================================================
	//	Methods
	//===============================================================

	
	public static void main(String[] args){
		Grid2D g = ImageUtil.wrapImageProcessor(IJ.openImage("D:\\Data\\Lena.tif").getProcessor());
		BurtPyramid pyramid = new BurtPyramid(g,3);
		Grid2D firstLevel = pyramid.getAtOctaveAndLevel(1, 2);
		new ImageJ();
		g.show("Original Image");
		firstLevel.show("Pyramid Image");
	}
	
	/**
	 * Constructor for precomputation of Pyramid levels.
	 * @param nOctaves
	 * @param lvlPerOctave
	 * @param img
	 */
	public BurtPyramid(int nOctaves, int lvlPerOctave, Grid2D img){
		init(nOctaves, lvlPerOctave, img);
		precompute();
	}
	
	/**
	 * Precomputes the octaves and levels of the requested Burt pyramid and stores the grids in the respective
	 * class member. This causes overhead, as it is called in the constructor directly.
	 */
	private void precompute(){
		this.precomp = new ArrayList<Grid2D[]>();
		for(int i = 0; i <= nOctaves; i++){
			Grid2D[] grids = new Grid2D[lvlPerOctave];
			for(int j = 1; j <= lvlPerOctave; j++){
				grids[j-1] = getAtOctaveAndLevel(i, j);
			}
			precomp.add(grids);
		}
		this.PRECOMPUTED = true;
	}
	
	/**
	 * Constructor that does not pre-compute the octaves and levels. 
	 * @param img
	 * @param lvlPerOctave Default is 1
	 */
	public BurtPyramid(Grid2D img, int lvlPerOctave){
		init(0, lvlPerOctave, img);
	}
	/**
	 * Constructor that does not pre-compute the octaves and levels. 
	 * @param img
	 */
	public BurtPyramid(Grid2D img){
		init(0, 1, img);
	}
	
	/**
	 * Initializes class members.
	 * @param nOctaves
	 * @param lvlPerOctave
	 * @param img
	 */
	private void init(int nOctaves, int lvlPerOctave, Grid2D img){
		this.nOctaves = nOctaves;
		this.lvlPerOctave = lvlPerOctave;
		this.level0 = ImageUtil.wrapGrid(img, "");
	}
	
	/**
	 * Calculates (or references if precomputed) the requested Grid at the specified octave.
	 * @param oct
	 * @return
	 */
	public Grid2D getAtOctave(int oct){
		return getAtOctaveAndLevel(oct, 1);
	}
	
	/**
	 * Calculates (or references if precomputed) the requested Grid at the specified octave and level.
	 * @param oct
	 * @param lvl
	 * @return
	 */
	public Grid2D getAtOctaveAndLevel(int oct, int lvl){
		assert(lvl != 0 && lvl <= lvlPerOctave) : new IllegalArgumentException("Level requested not valid.");
		
		if(PRECOMPUTED){
			assert(oct < nOctaves) : new IllegalArgumentException("Octave requested not valid.");
			return precomp.get(oct)[lvl-1];
		}
		
		ImageProcessor blurred = level0.getProcessor().duplicate().convertToFloat();
		GaussianBlur gs = new GaussianBlur();
			
		if(oct != 0){
			double sigmaOct = Math.pow(2,oct);
			gs.blurGaussian(blurred, sigmaOct, sigmaOct, 0.0002);
		}
			
		int widthOct = (int)(level0.getWidth()/(oct+1.0));
		int heightOct = (int)(level0.getHeight()/(oct+1.0));
		blurred = blurred.resize(widthOct, heightOct);
		//TODO level einbauen
		double sigmaLvl = (lvl-1.0)/lvlPerOctave;
		if(sigmaLvl != 0){
			gs.blurGaussian(blurred, sigmaLvl, sigmaLvl, 0.0002);
		}
		
		return ImageUtil.wrapImageProcessor(blurred);		
	}
	

}

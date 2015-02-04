package edu.stanford.rsl.conrad.phantom.renderer;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Class which generates an intensity ramp across the image in each image of the stack. Useful to investigate intensity-dependent processing.
 * 
 * @author akmaier
 *
 */
public class RampProjectionPhantomRenderer extends PhantomRenderer {

	private int width = 620;
	private int height = 480;
	private int stack = 191;
	private double slope = 0.008;
	private int projectionNumber;
	
	private Grid2D createFloatProcessor(int k){
		Grid2D fl = new Grid2D(width, height);
		double middle = ((width * slope) / 2) + 0.2;
		double currentSlope = slope * (((1.0 * k) - (this.stack / 2)) / (this.stack / 2));
		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++) {
				fl.putPixelValue(i, j, middle + ((i - (width / 2)) * currentSlope));
			}
		}
		return fl;
	}
	
	@Override
	public void createPhantom(){

	}

	/**
	 * sets the width of the image to be rendered
	 * @param width the image width
	 */
	public void setWidth(int width) {
		this.width = width;
	}

	/**
	 * returns the width
	 * @return the width
	 */
	public int getWidth() {
		return width;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getHeight() {
		return height;
	}

	public void setStack(int stack) {
		this.stack = stack;
	}

	public int getStack() {
		return stack;
	}

	public void setSlope(double slope) {
		this.slope = slope;
	}

	public double getSlope() {
		return slope;
	}

	public String toString(){
		return "Ramp Projection Phantom";
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public void configure() throws Exception {
		try {
			width = UserUtil.queryInt("Enter width", width);
			height = UserUtil.queryInt("Enter height:", height);
			stack = UserUtil.queryInt("Enter number of projection images", stack);
			slope = UserUtil.queryDouble("Enter slope: ", slope);
			configured = true;
		} catch (Exception e){
			System.out.println(e.getLocalizedMessage());
		}
	}

	@Override
	public Grid2D getNextProjection() {
		Grid2D proc  = null;
		if (projectionNumber < stack) {
			proc = createFloatProcessor(projectionNumber);
			projectionNumber ++;
		}
		return proc;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.workers;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.process.FloatProcessor;

/**
 * Phantom to create a Dirac pulse in the center of the projection. Useful to investigate frequency space.
 * 
 * 
 * @author akmaier
 *
 */
public class DiracProjectionPhantom extends SliceWorker {

	private double alpha = 0.1;

	@Override
	public void workOnSlice(int k){
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		int dimx = traj.getReconDimensionX();
		int dimy = traj.getReconDimensionY();
		FloatProcessor fl = new FloatProcessor(dimx, dimy);
		int width = fl.getWidth();
		int height = fl.getHeight();
		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++) {
				fl.putPixelValue(i, j, (1.0 - (Math.abs((0.0 + height / 2) - j) / (height / 2))) *  diracFunction(i - (width/2) - k, alpha));
			}
		}
		Grid2D grid = new Grid2D((float[])fl.getPixels(), fl.getWidth(), fl.getHeight());
		this.imageBuffer.add(grid, k);
	}

	/**
	 * Evaluates the delta function at x given alpha.
	 * @param x the x coordinate
	 * @param alpha the shape of the pulse
	 * @return value
	 */
	public static double diracFunction(double x, double alpha){
		return Math.exp(- (x * x) / (alpha * alpha)) / (alpha * Math.sqrt(Math.PI));
	}

	@Override
	public String toString(){
		return "Dirac Projection Phantom";
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@BOOK{Dirac58-TPO,\n" +
		"  author = {{Dirac}, P. A. M.},\n" +
		"  title = {{Principles of Quantum Mechanics}},\n" +
		"  publisher = {Oxford University Press Inc.},\n" +
		"  address = {New York, NY, United States},\n" +
		"  year = {1958}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Dirac PAM, Principles of Quantum Mechanics, Oxford University Press Inc., New York, NY, United States, 1958.";
	}

	@Override
	public void configure() throws Exception {
		alpha = UserUtil.queryDouble("Enter alpha: ", alpha);
		super.configure();
	}

	@Override
	public SliceWorker clone() {
		DiracProjectionPhantom clone = new DiracProjectionPhantom();
		clone.alpha = alpha;
		return clone;
	}


	@Override
	public String getProcessName() {
		return toString();
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
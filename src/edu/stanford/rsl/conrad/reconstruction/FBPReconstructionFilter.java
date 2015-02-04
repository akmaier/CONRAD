package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.Configuration;
/**
 * The FBP reconstruction filter is the abstract class that describes a general outline for a 
 * filtered backprojection algorithm.  It splits the reconstruction process into several 
 * backprojection steps which are called for each projection individually.  This is realized 
 * in the abstract method backproject.  The backprojection method wraps the backprojection of
 * a single projection into the reconstruction space.
 *  
 * @author akmaier
 *
 */
public abstract class FBPReconstructionFilter extends ReconstructionFilter {


	/**
	 * 
	 */
	private static final long serialVersionUID = 4294553655322598313L;


	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}
	
	/**
	 * Backprojects a single projection into the reconstruction space.
	 * @param projection the projection to backproject
	 * @param projectionNumber the number of the projection in the data set. This is used to identify the correct projection matrix.
	 * @throws Exception may happen.
	 */
	protected abstract void backproject(Grid2D projection, int projectionNumber) throws Exception;

	@Override
	protected void reconstruct() throws Exception {
		for (int i = 0; i < nImages; i++){
			try {
			backproject(inputQueue.get(i), i);
			} catch (Exception e){
				System.out.println("An error occured during backprojection of projection " + i);
			}
		}
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();
		int [] size = projectionVolume.getSize();
		
		for (int k = 0; k < size[2]; k++){
			sink.process(projectionVolume.getSubGrid(k), k);
		}
		sink.close();
		init = false;
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
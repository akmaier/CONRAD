package edu.stanford.rsl.conrad.filtering.multiprojection.blocks;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Class implements the processing for a straight forward 3D bilateral filter.
 * Implementation does not use any kind of speed ups. Use only with small values for width!
 * Otherwise the filter may take extremely long to finish the computation.
 * 
 * @author akmaier
 *
 */
public class IterativeReconstructionBlock extends ImageProcessingBlock {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7875829253535505054L;
	private double sigma_d = 2.0;
	private double sigma_r = 0.001;
	private int width = 5;

	@Override
	public ImageProcessingBlock clone() {
		IterativeReconstructionBlock clone = new IterativeReconstructionBlock();
		clone.sigma_d = sigma_d;
		clone.sigma_r = sigma_r;
		clone.width = width;
		return clone;
	}

	@Override
	protected void processImageBlock() {
		int[] size = inputBlock.getSize();
		// Create new grid as output block and fill it with zero values
		outputBlock = new Grid3D(size[0], size[1], size[2], true);
	}




	public void configure() throws Exception {
		width = UserUtil.queryInt("Enter Width: ", width);
		sigma_r = UserUtil.queryDouble("Sigma for photometric distance: ", sigma_r);
		sigma_d = UserUtil.queryDouble("Sigma for geometric distance: ", sigma_d);
		configured = true;
	}




}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
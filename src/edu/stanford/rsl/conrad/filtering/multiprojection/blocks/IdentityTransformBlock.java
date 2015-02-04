package edu.stanford.rsl.conrad.filtering.multiprojection.blocks;

public class IdentityTransformBlock extends ImageProcessingBlock {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6502352732903411093L;

	@Override
	public ImageProcessingBlock clone() {
		return new IdentityTransformBlock();
	}

	@Override
	protected void processImageBlock() {
		//inputBlock.show();
		outputBlock = inputBlock;
	}

	public void configure() throws Exception {
		configured = true;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
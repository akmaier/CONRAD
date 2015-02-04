package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.blocks.BilateralFilter3DBlock;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class IterativeReconstructionFilter extends BlockWiseMultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2430924392909104202L;

	@Override
	public ImageFilteringTool clone() {
		IterativeReconstructionFilter clone = new IterativeReconstructionFilter();
		clone.modelBlock = modelBlock;
		clone.numBlocks = numBlocks;
		clone.blockOverlap = blockOverlap;
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Iterative Reconstruction Filter";
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}
	
	@Override
	public void configure() throws Exception{
		numBlocks = 1;
		blockOverlap = new int[3];
		if (modelBlock == null) {
			modelBlock = new BilateralFilter3DBlock();
		}
		modelBlock.configure();
		configured = true;
	}
	

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.blocks.AnisotropicStructureTensorBlock;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class BlockWiseStructureTensor extends BlockWiseMultiProjectionFilter {



	/**
	 * 
	 */
	private static final long serialVersionUID = -5932376637078023191L;


	@Override
	public ImageFilteringTool clone() {
		BlockWiseStructureTensor clone = new BlockWiseStructureTensor();
		clone.modelBlock = modelBlock;
		clone.numBlocks = numBlocks;
		clone.configured = configured;
		clone.blockOverlap = blockOverlap;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Block-wise Anisotropic Structure Tensor-based Noise Filter";
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
		super.configure();
		if (modelBlock == null) {
			modelBlock = new AnisotropicStructureTensorBlock();
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
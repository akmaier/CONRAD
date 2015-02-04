package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.blocks.IdentityTransformBlock;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Class to test the BlockWiseMultiProjectionFilter. Input equals output.
 * 
 * @author akmaier
 *
 */
public class IdentityTransformFilter extends BlockWiseMultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6288980025231747250L;

	@Override
	public ImageFilteringTool clone() {
		return new IdentityTransformFilter();
	}

	@Override
	public String getToolName() {
		return "Identity Transform Filter";
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
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}
	
	@Override
	public void configure() throws Exception{
		super.configure();
		modelBlock = new IdentityTransformBlock();
		configured = true;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
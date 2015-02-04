package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * A simple approach for iterative reconstruction. Currently the method bodies are not implemented yet.
 * @author akmaier
 *
 */
public class SimpleIterativeReconstruction extends ReconstructionFilter {

	
	private static final long serialVersionUID = 7404570438691680164L;

	
	@Override
	protected void reconstruct() throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	@Override
	public String getToolName() {
		return "Simple Iterative Reconstruction";
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public void configure() throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
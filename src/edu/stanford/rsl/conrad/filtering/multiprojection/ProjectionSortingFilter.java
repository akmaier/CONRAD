package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class ProjectionSortingFilter extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1767982087589043403L;

	/**
	 * Creates a new ProjectionSortingFilter. As this one is a trivial filter it is already configured at construction.
	 */
	public ProjectionSortingFilter(){
		context = 0;
		configured = true;
	}
	


	@Override
	protected synchronized void processProjectionData(int projectionNumber) throws Exception {
		if (debug > 0) System.out.println("ProjectionSortingFilter: Projecting " + projectionNumber);
		sink.process(inputQueue.get(projectionNumber), projectionNumber);
		inputQueue.remove(projectionNumber);
	}

	@Override
	public ImageFilteringTool clone() {
		return new ProjectionSortingFilter();
	}

	@Override
	public String getToolName() {
		return "Projection Sorting Multi Projection Filter";
	}

	@Override
	public boolean isDeviceDependent() {
		// surely not.
		return false;
	}

	@Override
	public void configure() throws Exception {
		// not much to do here.
	}

	@Override
	public String getBibtexCitation() {
		// Who we're gonn'a call?
		return CONRAD.CONRADBibtex;
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
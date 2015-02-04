package edu.stanford.rsl.conrad.pipeline;

import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

/**
 * Interface to model an arbitrary projection source
 * 
 * @author akmaier
 *
 */
public interface ProjectionSource {

	/**
	 * Initializes the the ProjectionSource. This can either be a filename or a String that describes how to initialize the source.
	 * 
	 * @param filename the name of the file to read
	 * @throws IOException may happen
	 */
	public void initStream(String filename) throws IOException;
	
	/**
	 * Writes the next projection into an IndividualImagePipelineFiltering tool, i.e. reads the projection and sets the right projection number in a synchornized manner.
	 * <BR>
	 * @param tool the tool to write to
	 */
	public void getNextProjection(IndividualImagePipelineFilteringTool tool);
	
	/**
	 * Returns the current projection number
	 * @return the number
	 * 
	 * @see #getNextProjection()
	 * @see #getNextProjection(IndividualImagePipelineFilteringTool tool)
	 */
	public int getCurrentProjectionNumber();
	
	/**
	 * Returns the next projection. Note that this call may be out of sync with getCurrentProjectionNumber()
	 * @return the projection
	 * 
	 * @see #getCurrentProjectionNumber()
	 * @see #getNextProjection(IndividualImagePipelineFilteringTool tool)
	 */
	public Grid2D getNextProjection();
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
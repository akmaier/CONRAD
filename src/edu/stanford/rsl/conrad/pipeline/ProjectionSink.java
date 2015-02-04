package edu.stanford.rsl.conrad.pipeline;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;


/**
 * Interface for an ImageProcessor Output Stream. Note that the streams also have a name.
 * @author akmaier
 *
 */
public interface ProjectionSink {
	public void process(Grid2D projection, int projectionNumber) throws Exception;
	public abstract String getName();
	public void setShowStatus(boolean showStatus);
	public void close() throws Exception;
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
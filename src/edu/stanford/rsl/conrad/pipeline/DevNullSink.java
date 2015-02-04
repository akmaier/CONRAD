package edu.stanford.rsl.conrad.pipeline;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;


/**
 * Class for debugging pipelines. Will discard any information sent to this sink.
 * 
 * @author akmaier
 *
 */
public class DevNullSink extends BufferedProjectionSink {

	public DevNullSink(){
		configured = true;
	}
	/**
	 * 
	 */
	private static final long serialVersionUID = -6582927447254656308L;

	@Override
	public void close() throws Exception {
		
	}

	@Override
	public void configure() throws Exception {
		
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
	public void process(Grid2D projection, int projectionNumber)
			throws Exception {
		
	}

	@Override
	public String getName() {
		return "/dev/null";
	}

	@Override
	public void setConfiguration(Configuration config) {
		
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
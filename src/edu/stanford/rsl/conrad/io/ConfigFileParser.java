package edu.stanford.rsl.conrad.io;

import java.io.IOException;

/**
 * Interface for classes which parse config files.
 * 
 * @author akmaier
 *
 */
public interface ConfigFileParser {
	/**
	 * Reads the configuration from the file denoted by filename
	 * @param filename the file name
	 * @throws IOException may happen during reading
	 */
	public void readConfigFile(String filename) throws IOException;
	
	/**
	 * is true if the reading was successful.
	 * @return reading success?
	 */
	public boolean getSuccess();
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
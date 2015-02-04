package edu.stanford.rsl.apps.gui;

/**
 * Interface to model configuration via a graphical user interface.
 * @author akmaier
 *
 */
public interface GUIConfigurable {
	/**
	 * Configures the object before execution
	 * @throws Exception may happen
	 */
	public void configure() throws Exception;
	
	/**
	 * Is true if the object was successfully configured
	 * @return configured?
	 */
	public boolean isConfigured();
}


/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
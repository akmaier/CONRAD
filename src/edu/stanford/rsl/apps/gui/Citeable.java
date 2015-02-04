package edu.stanford.rsl.apps.gui;

/**
 * Interface to model citeable objects
 * @author akmaier
 *
 */
public interface Citeable {
	
	/**
	 * Returns the citation in bibTex format
	 * @return citation as String
	 */
	public String getBibtexCitation();
	
	/**
	 * Returns the citation in Medline format
	 * @return citation as String
	 */
	public String getMedlineCitation();
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
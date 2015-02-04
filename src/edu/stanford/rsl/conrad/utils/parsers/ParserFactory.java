package edu.stanford.rsl.conrad.utils.parsers;

import java.io.File;

import edu.stanford.rsl.conrad.phantom.forbild.ForbildParser;


/**
 * Retrieves appropriate  phaser given a filename
 * 
 * @author Rotimi X Ojo 
 */
public class ParserFactory {
	
	public static SceneFileParser getSceneFileParser(String filename){
		if(filename.substring(filename.indexOf(".")).toLowerCase().equals(".pha")){
			return new ForbildParser(new File(filename));
		}
		return new ForbildParser(new File(filename));
	}

}
/*
 * Copyright (C) 2010-2014  Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials.database.xmldatagenerators;


/**
 * Class to build a xml database of defined materials (Elements, Compounds, Mixtures).
 * To be used incase of data corruption
 * @author Rotimi X Ojo
 *
 */
public class RebuildDatabase {

	
	public static void main(String[] args) throws Exception{
		RebuildMaps.main(null);
		RebuildMaterialDatabase.main(null);		
	}

	

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
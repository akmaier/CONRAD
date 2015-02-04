package edu.stanford.rsl.conrad.physics.materials.database;

import java.io.File;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.utils.XmlUtils;

/**
 * Persistent and Extensible Map from material name to formula.
 * @author Rotimi X Ojo
 */
public class NameToFormulaMap {
	
	public static final File file = new File(MaterialsDB.getDatabaseLocation() + "configfiles/maps/nameToFormulaMap.xml");
	@SuppressWarnings("unchecked")
	public static final TreeMap<String, String> map = (TreeMap<String, String>) XmlUtils.deserializeObject(file);
	
	/**
	 * Retrieves the material formula corresponding to the given name
	 * @param name is name of material
	 * @return null if name is not found.
	 */
	public static String getFormula(String name) {
		return map.get(name);
	}
	
	/**
	 * Associates a given name to a formula. if name has already been mapped to a formula, the previous association is replaced.
	 * @param name is name of material
	 * @param formula is formula of material	 
	 */
	public static void put(String name, String formula){
		map.put(name, formula);
		XmlUtils.serializeObject(file, map);
	}
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
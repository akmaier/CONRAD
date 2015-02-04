package edu.stanford.rsl.conrad.physics.materials.database;

import java.io.File;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.utils.XmlUtils;

/**
 * Persistent and Extensible Map from material formula to material name.
 * @author Rotimi X Ojo
 */
public class FormulaToNameMap {
	public static final File file = new File(MaterialsDB.getDatabaseLocation() + "configfiles/maps/formulaToNameMap.xml");
	@SuppressWarnings("unchecked")
	public static final TreeMap<String, String> map = (TreeMap<String, String>) XmlUtils.deserializeObject(file);
	
	/**
	 * Retrieves the material name corresponding to the given formula
	 * @param formula is formula of material
	 * @return null if formula is not found.
	 */
	public static String getName(String formula) {
		return map.get(formula);
	}
	
	/**
	 * Associates a given formula to a name. if formula has already been mapped to a name, the previous association is replaced.
	 * @param formula is formula of material
	 * @param name is name of material
	 */
	public static void put(String formula, String name){
		map.put(formula, name);
		XmlUtils.serializeObject(file, map);
	}
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
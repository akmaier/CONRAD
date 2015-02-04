package edu.stanford.rsl.conrad.physics.materials.database;

import java.io.File;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.utils.XmlUtils;
import edu.stanford.rsl.conrad.utils.interpolation.NumberInterpolatingTreeMap;

/**
 * Class to store and retrieve of mass attenuation data of elements. 
 * @author Rotimi X Ojo *
 */
public class ElementalMassAttenuationData {
	public static final String fileloc = MaterialsDB.getDatabaseLocation() + "configfiles/massAttenuationData/" ;
	
	/**
	 * Store mass attenuation data of element with given name
	 * @param elementName is name of element
	 * @param massAttData is mass attenuation data of element
	 */
	public static  void put(String elementName, TreeMap<AttenuationType, NumberInterpolatingTreeMap> massAttData){
		XmlUtils.serializeObject(new File(fileloc + elementName.toLowerCase() + ".xml"), massAttData);
	}
	
	/**
	 * Retrieve mass attenuation data of given element
	 * @param elementName is name of element of interest
	 * @return mass attenuation data of given element.
	 */
	@SuppressWarnings("unchecked")
	public static TreeMap<AttenuationType, NumberInterpolatingTreeMap> get(String elementName){
		return (TreeMap<AttenuationType, NumberInterpolatingTreeMap>) XmlUtils.deserializeObject(new File(fileloc + elementName.toLowerCase() + ".xml"));
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials.utils;

import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.NoSuchElementException;

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * <p>This class contains methods for generating custom materials, calculating energy dependent mass attenuation, 
 * <br/>and determining the energy dependent attenuation coefficient of a material given its density, and energy dependent mass attenuation.<p>
 * 
 * @author Rotimi X Ojo, Andreas Maier
 */
public class MaterialUtils {	

	/**
	 * Computes the total (molar) mass of the weighted atomic composition.
	 * @param wac the weighted composition
	 * @return the (molar) mass
	 */
	public static double computeMolarMass(WeightedAtomicComposition wac){
		double sum =0;
		Iterator<Double> iter = wac.valuesIterator();
		while(iter.hasNext()){
			sum += iter.next();
		}
		return sum;
	}
	
	/**
	 * Generates a new instance of material with given characteristics
	 * @param name is  material identifier
	 * @param density is material density
	 * @param composition is atomic composition by weight of material
	 * @return null if material cannot be created
	 */
	public static Material newMaterial(String name, double density, WeightedAtomicComposition composition){
		Material material  = new Material();
		material.setName(name);
		material.setDensity(density);
		material.setWeightedAtomicComposition(composition);
		return material;
	}	


	
	/**
	 * Convert an array of doubles to semi-colon separated values
	 * @param vals is array of double
	 * @return an array of doubles to semi-colon separated values
	 */
	public static String getArrayAsString(double[] vals) {
		NumberFormat f = NumberFormat.getInstance();  
		f.setGroupingUsed(false); 
		String arrayString = "";
		for(int i = 0; i < vals.length;i++){
			arrayString+=f.format(vals[i]) + ";";
		}
		return arrayString.trim();
	}
	
	public static HashMap <Material, double[]> loadAttenuationCoefficients (double energies [], AttenuationType att){
		HashMap <Material, double[]> attenuationCoefficientsMap = new HashMap<Material, double[]>();
		for (String materialString:MaterialsDB.getMaterials()){
			Material mat = MaterialsDB.getMaterial(materialString);
			updateAttenuationCoefficientsMap(attenuationCoefficientsMap, energies, mat, att);
		}
		return attenuationCoefficientsMap;
	}
	
	public static synchronized void updateAttenuationCoefficientsMap(HashMap <Material, double[]> attenuationCoefficientsMap, double [] energies, Material mat, AttenuationType att){
		double [] attenuation = new double[energies.length];
		try{
			for (int j=0; j<attenuation.length;j++){
				attenuation[j] = mat.getAttenuation(energies[j], att); 
			}
			attenuationCoefficientsMap.put(mat, attenuation);
		} catch (NoSuchElementException e) {
			System.out.println("Skipping " + mat + " as attenuation data is incomplete for the configured energies.");
		} catch (NullPointerException e2){
			System.out.println("Skipping " + mat + " as attenuation data is incomplete.");
		}
	}
	
	
}

/*
 * Copyright (C) 2010-2014 Rotimi X Ojo, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials.utils;

import java.util.Iterator;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.materials.database.ElementalMassAttenuationData;
import edu.stanford.rsl.conrad.physics.materials.database.FormulaToNameMap;
import edu.stanford.rsl.conrad.utils.interpolation.NumberInterpolatingTreeMap;

/**
 * Calculates the mass attenuation coefficient of a material given its formula or weighted atomic composition.
 * @see WeightedAtomicComposition
 * @author Rotimi X Ojo
 */
public class LocalMassAttenuationCalculator {
	
	private static final TreeMap<String, TreeMap<AttenuationType, NumberInterpolatingTreeMap> > cache = new TreeMap<String, TreeMap<AttenuationType,NumberInterpolatingTreeMap>>();
	
	/**
	 * Calculates the mass attenuation coefficient of a material given its formula
	 * @param formula is the chemical formula of the material
	 * @param energy is the energy of interest
	 * @param attType is the {@link AttenuationType} of interest
	 * @return the mass attenuation coefficient of a material given its formula
	 */
	public static double getMassAttenuationData(String formula,
			double energy, AttenuationType attType) {
		return getMassAttenuationData(new WeightedAtomicComposition(formula), energy, attType);
	}
	
	/**
	 * Calculates the mass attenuation coefficient of a material given its formula
	 * @param formula is the chemical formula of the material
	 * @param energies is the set of energy of interest
	 * @param attType is the {@link AttenuationType} of interest
	 * @return the mass attenuation coefficient of a material given its formula
	 */
	public static NumberInterpolatingTreeMap getMassAttenuationData(String formula,
			double [] energies, AttenuationType attType) {
		NumberInterpolatingTreeMap num = new NumberInterpolatingTreeMap();
		for(double energy:energies){
			num.put(energy, getMassAttenuationData(new WeightedAtomicComposition(formula), energy, attType));
		}
		return num;
	}
	
	/**
	 * Calculates the mass attenuation coefficient of a material given its {@link WeightedAtomicComposition}
	 * @param comp is the weighted atomic composition of the material
	 * @param energies is the set of energy of interest
	 * @param attType is the {@link AttenuationType} of interest
	 * @return the mass attenuation coefficient of a material given its formula
	 */
	public static NumberInterpolatingTreeMap getMassAttenuationData(WeightedAtomicComposition comp,
			double [] energies, AttenuationType attType) {
		NumberInterpolatingTreeMap num = new NumberInterpolatingTreeMap();
		for(double energy:energies){
			num.put(energy, getMassAttenuationData(comp, energy, attType));
		}
		return num;
	}
	
	/**
	 * Calculates the mass attenuation coefficient of a material given its weighted atomic composition.
	 * This method uses an internal static cache to inhibit excessive file access. However, this
	 * requires the method to be synchronized. This may lead to quite slow performance in parallel use.
	 * 
	 * @param comp is the {@link WeightedAtomicComposition} of the material
	 * @param energy is the energy of interest
	 * @param attType is the {@link AttenuationType} of interest
	 * @return the mass attenuation coefficient of a material given its weighted atomic composition.
	 */
	public static synchronized double getMassAttenuationData(WeightedAtomicComposition comp,
			double energy, AttenuationType attType) {
		SimpleVector vec = getNormalizedComposition(comp);
		Iterator<String> it = comp.keysIterator();
		double value = 0;
		int counter = 0;
		while(it.hasNext()){
			TreeMap<AttenuationType, NumberInterpolatingTreeMap> massAtt = null;
			String name = FormulaToNameMap.getName(it.next());
			if((massAtt = cache.get(name))== null){
				massAtt= ElementalMassAttenuationData.get(name);
				cache.put(name, massAtt);
			}
			double buff = massAtt.get(attType).interpolateValue(energy).doubleValue();
			value+=vec.getElement(counter)* buff;
			counter++;
		}
		return value;
	}

	private static SimpleVector getNormalizedComposition(WeightedAtomicComposition comp) {
		SimpleVector vec = new SimpleVector(comp.size());
		Iterator<Double> it = comp.valuesIterator();
		int i =0;
		double sum = 0;
		while(it.hasNext()){
			vec.setElementValue(i, it.next());
			sum+=vec.getElement(i);
			i++;
		}
		vec.divideBy(sum);
		return vec;
	}





}

/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials.utils;

import java.io.Serializable;
import java.util.Iterator;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.physics.materials.Element;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * <p>This class stores the weighted atomic composition of a composite material.
 * <br/>The weighted atomic composition of a material is a key-sorted map containing its constituting elements(keys) and their contribution by weight.</p>
 * <p>H2O is stored as:</p>
 * <p><H , 2 * atomic weight of hydrogen><br/>
 * <O , 1 * atomic weight of oxygen></p>
 * 
 * @author Rotimi .X. Ojo
 *
 */
public class WeightedAtomicComposition implements Serializable {
/**
	 * 
	 */
	private static final long serialVersionUID = 3049665041572231953L;
	private TreeMap<String, Double> composition  = new TreeMap<String, Double>();
	
/**
 * Create an empty weighted atomic composition table.
 */
	public WeightedAtomicComposition(){}
	
	/**
	 * Create a new weighted atomic composition table by parsing formula and determining the atomic composition by weight. This constructor is generally used if the element of the material is entirely defined by formula.
	 * @param formula is formula of material
	 */
	public WeightedAtomicComposition(String formula){
		add(formula, 1);
	}
	
	/**
	 * Create a new weighted atomic composition table by parsing formula and determining the atomic composition by weight scaled by proportion. This constructor is generally used to initialize composite materials.
	 * @param formula  is formula of a constituent of the material to be described by composition table
	 * @param proportion is proportion by weight of the material defined by formula
	 */
	public WeightedAtomicComposition(String formula, double proportion){
		add(formula, proportion);
	}	
	
	/**
	 * Add an element to table.
	 * @param formula is formula of element
	 * @param proportion is atomic contribution by weight. For a compound, proportion = number of atoms * atomic weight;
	 */
	public void add(String formula, double proportion){
		int lastIndex = 0;
		for(int i = 1; i < formula.length(); i++){
			String curr = formula.substring(i,i+1);
			if(curr.toUpperCase().equals(curr) && !curr.matches("\\p{Digit}+")){					
				addTokenToMap(formula.substring(lastIndex, i), proportion);			
				lastIndex = i;
			}
		}
		addTokenToMap(formula.substring(lastIndex).trim(), proportion);
	}	
	
	/**
	 * <p>To be used if there exists only one instance of element in formula. This method by rebuildDatabase for bootstrapping.</div>
	 * <br>Users are encouraged to use other methods.</p>
	 * @param formula is formula of element
	 * @param weight is atomic weight of element
	 */
	public void addUniqueElement(String formula, double weight){
		composition.put(formula.trim(), weight);
	}
	
	
	private void addTokenToMap(String token, double proportion) {
		int stop = indexOfDigit(token);	
		String element = token;
		double weight = 0;
		if(stop == -1){
			weight = ((Element)MaterialsDB.getMaterial(token)).getAtomicWeight()*proportion;
		}else{
			element = token.substring(0, stop).trim();
			weight = Double.valueOf(token.substring(stop))*((Element)MaterialsDB.getMaterial(element)).getAtomicWeight()*proportion;
		}
		if (composition.containsKey(element)) {
			weight += composition.get(element);
		}
		composition.put(element, weight);
	}	
	
	

	private int indexOfDigit(String token) {
		for(int i = 0; i < token.length(); i++){
			String Char = token.charAt(i)+"";
			if(Char.matches("\\p{Digit}+")){
				return i;
			}
		}
		return -1;
	}
	
	/**
	 * Set a pre-initialized composition table
	 * @param composition is composition table describing the elemental makeup of the material of interest
	 */
	public void  setCompositionTable(TreeMap<String, Double> composition) {
		this.composition = composition;
	}

	/**
	 * Retrieve a composition table describing the elemental make up of material of interest
	 * @return the composition table describing the elemental make up of material of interest
	 */
	public TreeMap<String, Double> getCompositionTable() {
		return composition;
	}

	/**
	 * Returns the number of elements that constitute the material of interest
	 * @return a positive integer representing the number of elements that constitute the material of interest
	 */
	public int size() {
		return composition.size();
	}
	
	/**
	 * Iterate through the composition of elements constituting the material of interest
	 * @return the composition of elements constituting the material of interest
	 */
	public Iterator<Double> valuesIterator() {
		return composition.values().iterator();
	}
	
	/**
	 * Iterate through the elements constituting the material of interest
	 * @return the elements constituting the material of interest
	 */
	public Iterator<String> keysIterator(){
		return composition.keySet().iterator();
	}

}

package edu.stanford.rsl.conrad.physics.materials;

import java.util.Iterator;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * <p>This class stores the weighted atomic composition of a material.
 * <br/>The weighted atomic composition of a material is a key-sorted map containing its constituting elements(keys) and their contribution by weight.</p>
 * <p>H2O is stored as:</p>
 * <p><H , 2 * atomic weight of hydrogen><br/>
 * <O , 1 * atomic weight of oxygen></p>
 * 
 * @author Rotimi .X. Ojo
 *
 */
public class WeightedAtomicComposition {
private TreeMap<String, Double> composition  = new TreeMap<String, Double>();
	
	public WeightedAtomicComposition(){}
	
	
	public WeightedAtomicComposition(String formula){
		add(formula, 1);
	}
	
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
	 * <br>Users are encouraged to use the add method.</p>
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
	
	public void  setCompositionTable(TreeMap<String, Double> composition) {
		this.composition = composition;
	}


	public TreeMap<String, Double> getCompositionTable() {
		return composition;
	}


	public int size() {
		return composition.size();
	}


	public Iterator<Double> valuesIterator() {
		return composition.values().iterator();
	}
	
	public Iterator<String> keysIterator(){
		return composition.keySet().iterator();
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
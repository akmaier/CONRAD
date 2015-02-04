package edu.stanford.rsl.conrad.physics.materials;


/**
 * This class models an elemental material. 
 * @author Rotimi X Ojo
 *
 */
public class Element extends Material {

	private static final long serialVersionUID = -761247784943296794L;
	private String symbol = "";
	private double atomicWeight = 0;
	private double atomicNumber = 0;
	
	
	
	/**
	 * Retrieve the atomic symbol of the element
	 * @return the atomic symbol
	 */
	public String getSymbol(){
		return symbol;
	}
	
	/**
	 * Set the atomic symbol of the element.
	 * @param symbol is the atomic symbol of the element
	 */
	public void setSymbol(String symbol){
		this.symbol = symbol;	
	}
	
	/**
	 * Set the atomic number of the element
	 * @param atomicNumber
	 */
	public void setAtomicNumber(double atomicNumber) {
		this.atomicNumber = atomicNumber;
	}
	
	/**
	 * Retrieve the atomic number of the element.
	 * @return the atomic number of the element
	 */
	public double getAtomicNumber() {
		return atomicNumber;
	}
	
	/**
	 * Set the atomic weight of the element
	 * @param atomicWeight is the atomic weight of the element
	 */
	public void setAtomicWeight(double atomicWeight) {
		this.atomicWeight = atomicWeight;
	}
	
	/**
	 * Retrieve the atomic weight of the element.
	 * @return the atomic weight of the element
	 */
	public double getAtomicWeight() {
		return atomicWeight;
	}	
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials;


/**
 * Class to model a compound
 * @author Rotimi .X.Ojo
 *
 */
public class Compound extends Material {

	private static final long serialVersionUID = -5197674861970729455L;
	private String formula;

	/**
	 * @return the formula of a compound
	 */
	public String getFormula(){
		return formula;
	}
	
	/**
	 * Set the formula of a compound
	 * @param formula id formula of initialized compound
	 */
	public void setFormula(String formula){
		this.formula = formula;	
	}	

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials.utils;

/**
 * Different types of attenuation.
 * 
 * @author Rotimi X Ojo
 *
 */
public enum AttenuationType {
	
	COHERENT_ATTENUATION("coherent"),
	INCOHERENT_ATTENUATION("incoherent"),
	PHOTOELECTRIC_ABSORPTION("photoelectric"),
	NUCLEAR_FIELD_PAIRPRODUCTION("nuclear"),
	ELECTRON_FIELD_PAIRPRODUCTION("electron"),		
	TOTAL_WITH_COHERENT_ATTENUATION("with"),
	TOTAL_WITHOUT_COHERENT_ATTENUATION("without");
	
	private String name;

	AttenuationType(String name){
		this.name = name;
		
	}
	
	public String getName(){
		return name;
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.materials;

import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.physics.Constants;
import edu.stanford.rsl.conrad.physics.materials.database.CompositionToAbsorptionEdgeMap;
import edu.stanford.rsl.conrad.physics.materials.database.OnlineMassAttenuationDB;
import edu.stanford.rsl.conrad.physics.materials.materialsTest.TestMassAttenuationData;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationRetrievalMode;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.physics.materials.utils.LocalMassAttenuationCalculator;
import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;

/**
 * <p>This class models an arbitrary material. Materials are defined by density and their characteristic attenuation of XRays of different energies.</p>
 * <p> Note. A material without a defined energy dependent attenuation cannot be used in polychromatic XRay projection modeling</p>.
 * @author Rotimi X Ojo
 */
public class Material implements Serializable, Cloneable{	
	
	private static final long serialVersionUID = -335499247531855579L;
	private String name = "";	
	private double density = -1;	
	private transient double CTValue = 0;
	private WeightedAtomicComposition comp;
	
	
	public Material(){}
	
	/**
	 * Initializes a material with a given density. To support polychromatic absorption, an energy dependent attenuation TreeMap must be provided.
	 * @param density is density of initialized material
	 */
	public Material(double density){
		setDensity(density);
	}
	
	public Material(Material material) {
		this.name = new String(material.name);
		this.density = material.density;
		this.CTValue = material.CTValue;
		// shallow copy of composition tree!
		this.comp = material.comp;
	}

	/**
	 * Update the name of the material.
	 * @param name is name of material
	 */	
	public void setName(String name) {
		this.name = name;
	}
	
	/**
	 * Retrieve the name of the material
	 * @return the name of the material
	 */
	public String getName() {
		return name;
	}
	
	/**
	 * Retrieve the CT value of a given material in [HU]
	 * @return 0 if CTValue is not available
	 */
	public double getCTValue() {
		return CTValue;
	}
	
	/**
	 * Retrieve the energy dependent attenuation of material
	 * @param energy is energy of interest in KeV
	 * @param attType is type of attenuation, all types of attenuation are supported.
	 * @return double representing the energy dependent attenuation of material sourced locally in [cm^-1].
	 * @see AttenuationType
	 * @see TestMassAttenuationData
	 */
	public double getAttenuation(double energy, AttenuationType attType) {
		return getAttenuation(energy, attType, AttenuationRetrievalMode.LOCAL_RETRIEVAL);
	}
	
	/**
	 * Retrieve the energy dependent attenuation of material. 
	 * @param energy is energy of interest in KeV
	 * @param attType is type of attenuation, all types of attenuation are supported.
	 * @param mode is retrieval mode.
	 * @return is type of attenuation, all types of attenuation are supported.
	 * @see AttenuationType
	 * @see AttenuationRetrievalMode
	 * @see TestMassAttenuationData
	 */
	public double getAttenuation(double energy, AttenuationType attType, AttenuationRetrievalMode mode){		
		if(mode.equals(AttenuationRetrievalMode.ONLINE_RETRIEVAL)){
			return density * OnlineMassAttenuationDB.getMassAttenuationData(comp, energy/1000, attType);
		}else{
			return density * LocalMassAttenuationCalculator.getMassAttenuationData(comp, energy/1000, attType);
		}
	}
	
	/**
	 * Update the density and CT Value of the material with given value
	 * @param density is density of the material
	 */
	public void setDensity(double density) {
		this.density = density;
		CTValue = Constants.computeCTValue(density); 
	}
	
	/**
	 * Retrieves the density of the material in [g/cm^3]
	 * @return -1 if density is not available
	 */	
	public double getDensity() {
		return density;
	}

	/**
	 * Update the atomic composition of material by mass
	 * @param comp is Treemap containing the atomic composition of material by mass
	 */
	
	public void setWeightedAtomicComposition(WeightedAtomicComposition comp){
		this.comp = comp;
	}
	
	/**
	 * Retrieve the atomic composition of material by mass
	 * @return null if atomic composition table is not available
	 */
	public WeightedAtomicComposition getWeightedAtomicComposition(){
		return comp;
	}
	
	/**
	 * Retrieve all the absorption edges of material in MeV.
	 * @return an empty array if there are no absorption edge
	 */
	public ArrayList<Double> getAbsorptionEdges(){
		return CompositionToAbsorptionEdgeMap.getAbsorptionEdges(comp);
	}	
	
	/**
	 * Materials are hashed by their name.
	 * @return the hash code of the material name.
	 */
	public int hashCode(){
		return getName().hashCode();
	}
	
	@Override
	public boolean equals(Object other){
		if (other instanceof Material)
			return getName().equals(((Material)other).getName());
		else 
			return false;
	}
	
	/**
	 * Materials are equal if they have the same name.
	 * @param other the other material
	 * @return true if name is the same.
	 */
	public boolean equals(Material other){
		return getName().equals(other.getName());
	}
	
	@Override
	public String toString(){
		return name;
	}
	
	@Override
	public Material clone(){
		return new Material(this);
	}
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
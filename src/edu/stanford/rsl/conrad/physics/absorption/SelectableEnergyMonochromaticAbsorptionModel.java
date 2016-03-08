/*
 * Copyright (C) 2014 Andreas Maier, Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.conrad.physics.absorption;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class SelectableEnergyMonochromaticAbsorptionModel extends
		AbsorptionModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7218141334431913465L;
	/**
	 * Energy [keV]
	 */
	double energy = 80;
	boolean configured = false;
	
	public SelectableEnergyMonochromaticAbsorptionModel(){
		super();
	}
	
	
	@Override
	public double evaluateLineIntegral(ArrayList<PhysicalObject> segments) {
		//outputSpectrum = new PolychromaticXRaySpectrum();
		double sum = 0;
		for (PhysicalObject o: segments){			
			double att = o.getMaterial().getAttenuation(energy,AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION);
			double len = ((Edge)o.getShape()).getLength();
			sum +=  att * len;
		}
		
		if(sum < 0){
			sum = 0;
		}
		
		// TODO: Normalization is never considered in the backprojectors, 
					// 		 thus, iteratively applying forward and backward projections
					//		 would yield to a scaling issue!
					//
					// length is in [mm]
					// attenuation is in [g/cm^3]
					// conversion from [g*mm/cm^3] = [g*0.1cm/cm^3] to [g/cm^2]
					// --> sum/10.0;
		return  sum / 10;
	}

	@Override
	public String toString() {
		if(configured) return "Monochromatic Absorption ("+energy+" keV)";
		else return "Monochromatic Absorption (select energy)";
	}


	@Override
	public void configure() throws Exception {
		try {
			energy=UserUtil.queryDouble("Enter energy in [keV]: ", energy);
			configured = true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void configure(double energy){
		this.energy=energy;
		configured = true;
	}


	@Override
	public boolean isConfigured() {
		return configured;
	}
	
	public void setEnergy(double energy) {
		this.energy = energy;
	}
	
	public double getEnergy() {
		return energy;
	}

	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
}

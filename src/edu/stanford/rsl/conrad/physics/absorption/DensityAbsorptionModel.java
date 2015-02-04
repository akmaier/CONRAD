/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.absorption;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Creates a absorption model for monochromatic projections
 * @author Rotimi X Ojo
 */
public class DensityAbsorptionModel extends AbsorptionModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7805745771273109739L;

	public double evaluateLineIntegral(ArrayList<PhysicalObject> segments) {
		double sum = 0;
		for (PhysicalObject o: segments){			
			double att = o.getMaterial().getDensity();
			double len = ((Edge)o.getShape()).getLength();
			sum +=  att * len;
		}

		// TODO: Normalization is never considered in the backprojectors, 
		// 		 thus, iteratively applying forward and backward projections
		//		 would yield to a scaling issue!
		//
		// length is in [mm]
		// attenuation is in [g/cm^3]
		// conversion from [g*mm/mc^3] = [g*0.1cm/cm^3] to [g/cm^2]
		// double t = sum/10.0;
		double t = sum;
		if(t < CONRAD.SMALL_VALUE){
			return 0;
		}
		return t;
	}

	@Override
	public String toString() {
		return "Density as Attenuation Model";
	}

	@Override
	public void configure() throws Exception {
		
	}

	@Override
	public boolean isConfigured() {
		return true;
	}
	
}

/*
 * Copyright (C) 2014 Andreas Maier, Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.absorption;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.StatisticsUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class SelectableEnergyMonochromaticAbsorptionModelWithNoise extends
		SelectableEnergyMonochromaticAbsorptionModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5967404911378538039L;
	double photonNumber = 50000;
	
	@Override
	public void configure() throws Exception {
		try {
			photonNumber=UserUtil.queryDouble("Enter Number of photons: ", photonNumber);
			super.configure();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	public String toString() {
		if(configured) return super.toString() + " with " + photonNumber + " photons";
		else return "Monochromatic Absorption (select energy) with noise";
	}
	
	@Override
	public double evaluateLineIntegral(ArrayList<PhysicalObject> segments) {
		double lineIntegral = super.evaluateLineIntegral(segments);
		double photonsAfterAbsorption = photonNumber * Math.exp(-lineIntegral);
		double withNoise = StatisticsUtil.poissonRandomNumber(photonsAfterAbsorption);
		double lineIntegralWithNoise = -Math.log(withNoise/photonNumber);
		if (Double.isInfinite(lineIntegralWithNoise)) lineIntegralWithNoise = CONRAD.BIG_VALUE;
		return lineIntegralWithNoise;
	}
	
	public void setPhotonNumber(double photonNumber) {
		this.photonNumber = photonNumber;
	}
	
	public double getPhotonNumber() {
		return photonNumber;
	}
	
}

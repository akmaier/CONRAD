package edu.stanford.rsl.conrad.physics;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.LinearInterpolatingDoubleArray;
import edu.stanford.rsl.jpop.SimpleFunctionOptimizer;
import edu.stanford.rsl.jpop.SimpleOptimizableFunction;


public class HalfValueLayerFunction implements SimpleOptimizableFunction {

	private double [] xRayEnergy;
	private double [] xRaySpectrum;	
	private LinearInterpolatingDoubleArray attenuationCoefficients1;
	private LinearInterpolatingDoubleArray attenuationCoefficients2;
	private LinearInterpolatingDoubleArray attenuationCoefficients3;
	public double optimalX;
	
	public HalfValueLayerFunction (double [] energies, double [] spectrum, 
			LinearInterpolatingDoubleArray lida1, 
			LinearInterpolatingDoubleArray lida2, 
			LinearInterpolatingDoubleArray lida3){
		this.xRayEnergy = energies; // energy KeV
		this.xRaySpectrum = spectrum; // photon count #
		this.attenuationCoefficients1 = lida1;
		this.attenuationCoefficients2 = lida2;
		this.attenuationCoefficients3 = lida3;
	}

	@Override
	public double evaluate(double x) {
		
		double sumN0 = 0;
		double sumN1 = 0;
		double coeff1; // Al
		double coeff2; // CsI converter
		double coeff3; // Air absorption
		double detectorEfficiency = 1;
		double xRayQuanta = 1;
		boolean isHVLMeasuredAtDetector = false;
		boolean isAirAbsorptionCounted = true;
		
		try {
			for (int i = 0; i< xRaySpectrum.length; i++){
				coeff1 = attenuationCoefficients1.getValue(xRayEnergy[i]/1000);
				coeff2 = attenuationCoefficients2.getValue(xRayEnergy[i]/1000);
				coeff3 = attenuationCoefficients3.getValue(xRayEnergy[i]/1000);
				
				if(isHVLMeasuredAtDetector) {
					// Detector flat panel efficiency, thickness = 200 [g/cm2]
					detectorEfficiency = 1 - Math.exp(-coeff2 * 200);						
				}
				
				if(isAirAbsorptionCounted) {
					xRayQuanta = XRaySpectrum.xrqpRe(coeff3);
				}				
				
				// Exposure (R) of the spectrum specified by the vector phi at energies specified by E (keV).
				sumN1 += xRaySpectrum[i] * Math.exp(-coeff1 * x) * detectorEfficiency * xRayEnergy[i] / xRayQuanta;
				sumN0 += xRaySpectrum[i] * detectorEfficiency * xRayEnergy[i] / xRayQuanta;		
				
				// System.out.println(xRayEnergy[i]+"\t"+coeff1+"\t"+coeff2+"\t"+coeff3);
			}	
		}catch (Exception e){
			e.printStackTrace();
		}
		
		// System.out.println(x +"\t"+ Math.abs(sumN0/2 - sumN1));	
		
		return Math.abs(sumN0/2 - sumN1);
	}
	
	public void runOptimization() {
		SimpleFunctionOptimizer optimizer = new SimpleFunctionOptimizer();
		optimizer.setLeftEndPoint(0);
		optimizer.setRightEndPoint(10);
		//optimizer.setInitialGuess(0.44);
		optimizer.setAbsoluteTolerance(CONRAD.DOUBLE_EPSILON);
		optimizer.setRelativeTolerance(CONRAD.DOUBLE_EPSILON);
		//optimizer.
		optimalX = optimizer.minimize(this);		
	}
	
	public double getOptimalX(){
		return optimalX;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
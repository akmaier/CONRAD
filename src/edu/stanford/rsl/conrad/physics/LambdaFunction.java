package edu.stanford.rsl.conrad.physics;

import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.conrad.parallel.ParallelizableRunnable;
import edu.stanford.rsl.jpop.SimpleFunctionOptimizer;
import edu.stanford.rsl.jpop.SimpleOptimizableFunction;

/**
 * Evaluates the lambda Function according to 
 * Joseph PM, Spital RD. �A method for correcting bone induced artifacts in computed tomography scanners�, JCAT 1978;2:100-108.
 * 
 * @author akmaier
 *
 */

public class LambdaFunction implements SimpleOptimizableFunction, ParallelizableRunnable {

	private CountDownLatch latch = null;
	private double [] xRaySpectrum;
	private double [] softAttenuationCoefficients;
	private double [] hardAttenuationCoefficients;
	private double waterCorrectedValue;
	private double passedHardMaterial;
	private double epsilon;
	private double optimalLambda;
	
	/**
	 * Constructor for the lambda function.
	 * @param xRaySpectrum the incident x-ray spectrum
	 * @param softAttenuationCoefficients the energy dependent attenuation coefficients in the soft material
	 * @param hardAttenuationCoefficients the energy dependent attenuation coefficients in the hard material
	 * @param waterCorrectedValue the observed water corrected value
	 * @param passedHardMaterial the total amount of hard material which was passed in the observation.
	 */
	LambdaFunction(double [] xRaySpectrum, double  [] softAttenuationCoefficients, double [] hardAttenuationCoefficients, double waterCorrectedValue, double passedHardMaterial){
		this.xRaySpectrum = xRaySpectrum;
		this.softAttenuationCoefficients = softAttenuationCoefficients;
		this.hardAttenuationCoefficients = hardAttenuationCoefficients;
		this.waterCorrectedValue = waterCorrectedValue;
		this.passedHardMaterial = passedHardMaterial;
		epsilon = 0;
		for (int i = 0; i< xRaySpectrum.length; i++){
			epsilon += xRaySpectrum[i] * Math.exp(-softAttenuationCoefficients[i]*waterCorrectedValue);
		}
	}
	
	/**
	 * Constructor for the lambda function.
	 * @param xRaySpectrum the incident x-ray spectrum
	 * @param softAttenuationCoefficients the energy dependent attenuation coefficients in the soft material
	 * @param hardAttenuationCoefficients the energy dependent attenuation coefficients in the hard material
	 * @param waterCorrectedValue the observed water corrected value
	 * @param passedHardMaterial the total amount of hard material which was passed in the observation.
	 * @param epsilon the epsilon factor, i.e. right side of the integral equation.
	 */
	LambdaFunction(double [] xRaySpectrum, double  [] softAttenuationCoefficients, double [] hardAttenuationCoefficients, double waterCorrectedValue, double passedHardMaterial, double epsilon){
		this.xRaySpectrum = xRaySpectrum;
		this.softAttenuationCoefficients = softAttenuationCoefficients;
		this.hardAttenuationCoefficients = hardAttenuationCoefficients;
		this.waterCorrectedValue = waterCorrectedValue;
		this.passedHardMaterial = passedHardMaterial;
		this.epsilon = epsilon;
	}
	
	@Override
	public double evaluate(double lambda) {
		double revan = 0;
		for (int i = 0; i< xRaySpectrum.length; i++){
			revan += xRaySpectrum[i] * Math.exp((-softAttenuationCoefficients[i]*waterCorrectedValue) + (softAttenuationCoefficients[i]*passedHardMaterial*lambda) - (hardAttenuationCoefficients[i]*passedHardMaterial));
		}
		return Math.abs(revan - epsilon);
	}

	@Override
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}

	@Override
	public void run() {
		SimpleFunctionOptimizer optimizer = new SimpleFunctionOptimizer();
		optimizer.setLeftEndPoint(-2.0);
		optimizer.setRightEndPoint(2.0);
		optimalLambda = optimizer.minimize(this);		
		latch.countDown();
	}

	/**
	 * Method to access the optimal lambda after optimization.
	 * @return the optimal value
	 */
	public double getOptimalLambda(){
		return optimalLambda;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
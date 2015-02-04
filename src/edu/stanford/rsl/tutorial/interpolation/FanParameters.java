/*
 * Copyright (C) 2014 Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.interpolation;

/**
 * This class is a container for fan beam parameters.
 *  
 * @author Marcel Pohlmann
 * 
 */

public class FanParameters {
	// Fan-Beam Parameter Container
	// params := [gammaM, maxT, deltaT, focalLength, maxBeta, deltaBeta]
	private double[] params;
	
	public FanParameters(double[] params){
		this.params = params.clone();
	}
	
	public void setGammaM(double gammaM){
		this.params[0] = gammaM;
	}
	
	public void setMaxT(double MaxT){
		this.params[1] = MaxT;
	}
	
	public void setDeltaT(double deltaT){
		this.params[2] = deltaT;
	}
	
	public void setFocalLength(double focalLength){
		this.params[3] = focalLength;
	}
	
	public void setMaxBeta(double maxBeta){
		this.params[4] = maxBeta;
	}
	
	public void setDeltaBeta(double deltaBeta){
		this.params[5] = deltaBeta;
	}
	
	public double getGammaM(){
		return this.params[0];
	}
	
	public double getMaxT(){
		return this.params[1];
	}
	
	public double getDeltaT(){
		return this.params[2];
	}
	
	public double getFocalLength(){
		return this.params[3];
	}
	
	public double getMaxBeta(){
		return this.params[4];
	}
	
	public double getDeltaBeta(){
		return this.params[5];
	}
}
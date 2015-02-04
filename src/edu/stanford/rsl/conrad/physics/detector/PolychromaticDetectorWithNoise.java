/*
 * Copyright (C) 2014 Andreas Maier, Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.absorption.PolychromaticAbsorptionModel;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class PolychromaticDetectorWithNoise extends XRayDetector {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2186375966931202433L;
	boolean configured = false;
	//PolychromaticAbsorptionModel model;
	boolean photonCounting = false;
	
	@Override
	public void configure(){
		try {
			photonCounting = UserUtil.queryBoolean("Do you want to simulate a photon counting detector?");
			ArrayList<Object> modelList = CONRAD.getInstancesFromConrad(PolychromaticAbsorptionModel.class);
			Object [] modelArray = new PolychromaticAbsorptionModel [modelList.size()];
			modelList.toArray(modelArray);
			model = (PolychromaticAbsorptionModel) UserUtil.chooseObject("Select noise-free model", "Model Selection", modelArray, (Object) modelArray[0]);
			model.configure();
			configured = true;
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public String toString(){
		String name = "Polychromatic X-Ray Detector with noise";
		if (model!= null) name+=" " + model.toString();
		return name;
	}

	@Override
	public boolean isConfigured(){
		return configured;
	}

	@Override
	public void writeToDetector(Grid2D grid, int x, int y, ArrayList<PhysicalObject> segments){
		double value = ((PolychromaticAbsorptionModel) model).computeIntensity(segments, ((PolychromaticAbsorptionModel) model).getMinimalEnergy(), ((PolychromaticAbsorptionModel) model).getMaximalEnergy(), true, !photonCounting);
		if (photonCounting){
			value /= ((PolychromaticAbsorptionModel) model).getTotalPhotonFlux();
		} else {
			value /= ((PolychromaticAbsorptionModel) model).getTotalIntensity();
		}
		//Due to the accumulation round off error, it is possible for output intensity to be marginally greater  
		//than input intensity in vacuum. We deal with this problem by bounding output intensity by the input 
		//intensity.
		if(value > 1){
			value = 1;
			//throw new RuntimeException("PolychromaticAbsorptionModel: numerical instability found.");
		}
		//End
		grid.putPixelValue(x, y,  -Math.log(value));
	}



	/**
	 * @return the photonCounting
	 */
	public boolean isPhotonCounting() {
		return photonCounting;
	}

	/**
	 * @param photonCounting the photonCounting to set
	 */
	public void setPhotonCounting(boolean photonCounting) {
		this.photonCounting = photonCounting;
	}

	/**
	 * @param configured the configured to set
	 */
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
}

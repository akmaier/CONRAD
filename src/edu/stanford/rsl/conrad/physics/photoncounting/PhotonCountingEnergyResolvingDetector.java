/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.photoncounting;

import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.absorption.PolychromaticAbsorptionModel;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class PhotonCountingEnergyResolvingDetector extends XRayDetector {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8932398731012572114L;
	public static final String I_ZERO_VECTOR = "I_ZERO_VECTOR";
	
	public static SimpleVector getIZeroVector (){
		return new SimpleVector(Configuration.getGlobalConfiguration().getRegistry().get(I_ZERO_VECTOR));
	}
	
	public static void setIZeroVector(SimpleVector vector){
		Configuration.getGlobalConfiguration().getRegistry().put(I_ZERO_VECTOR, vector.toString());
	}
	
	boolean noise;
	int numberofBins = 3;
	/**
	 * overlap in [keV] 
	 * Each energy bin is overlapping with the neighboring by this number.
	 */
	double overlap = 3;
	boolean photoncount = false;
	/**
	 * array of thresholds as double array in keV.
	 */
	double thresholds [];

	double photonFluxPerBin [];

	String [] channelNames;

	//PolychromaticAbsorptionModel model;

	boolean configured = false;

	@Override
	public void configure(){
		try {
			model = (PolychromaticAbsorptionModel) UserUtil.queryObject("Select base model", "Model Selection", PolychromaticAbsorptionModel.class);
			// we use this method to preload different material properties in order to be able to execute the code in parallel efficiently.
			// precompute energies:
			model.configure();
			// precompute intensities:
			numberofBins = UserUtil.queryInt("Enter number of bins:", numberofBins);
			overlap = UserUtil.queryDouble("Overlap in [keV]: ", overlap);
			thresholds = new double[numberofBins+1];
			for (int i = 0; i < thresholds.length; i++){
				thresholds[i]=UserUtil.queryDouble("Threshold " + i, ((PolychromaticAbsorptionModel)model).getMaximalEnergy() * ((double)i)/(numberofBins));
			}
			noise = UserUtil.queryBoolean("Simulate Noise?");
			photoncount = !UserUtil.queryBoolean("Apply -log?");
			photonFluxPerBin = new double[numberofBins];	
			for(int c = 0; c < numberofBins; c++){
				photonFluxPerBin[c] = ((PolychromaticAbsorptionModel)model).getPhotonFluxIntegral(thresholds[c], thresholds[c+1]);
			}
			SimpleVector iVector = new SimpleVector(photonFluxPerBin);
			setIZeroVector(iVector);
			
			channelNames = new String[numberofBins];
			for(int c = 0; c < numberofBins; c++){
				channelNames[c]="Channel I_" + c + ": " + photonFluxPerBin[c];
			}
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

	public String toString(){
		return "Photon Counting Energy Resolving Detector";
	}



	@Override
	public void writeToDetector(Grid2D grid, int x, int y, ArrayList<PhysicalObject> segments){
		for(int c = 0; c < numberofBins; c++){
			double intensity = ((PolychromaticAbsorptionModel)model).computeIntensity(segments, thresholds[c] - (overlap/2), thresholds[c+1]+(overlap/2), noise, false);
			if (!photoncount) {
				intensity /= photonFluxPerBin[c];
				if(intensity > 1){
					intensity = 1;
					//throw new RuntimeException("PolychromaticAbsorptionModel: numerical instability found.");
				}
				intensity = -Math.log(intensity);
			}
			((MultiChannelGrid2D)grid).putPixelValue(x, y, c, intensity);
		}
	}

	@Override
	public Grid2D createDetectorGrid(int width, int height){
		MultiChannelGrid2D grid=new MultiChannelGrid2D(width, height, numberofBins);
		grid.setChannelNames(channelNames);
		return grid;
	}

	/**
	 * @return the numberofBins
	 */
	public int getNumberofBins() {
		return numberofBins;
	}

	/**
	 * @param numberofBins the numberofBins to set
	 */
	public void setNumberofBins(int numberofBins) {
		this.numberofBins = numberofBins;
	}

	/**
	 * @return the thresholds
	 */
	public double[] getThresholds() {
		return thresholds;
	}

	/**
	 * @param thresholds the thresholds to set
	 */
	public void setThresholds(double[] thresholds) {
		this.thresholds = thresholds;
	}

	/**
	 * @return the configured
	 */
	public boolean isConfigured() {
		return configured;
	}

	/**
	 * @param configured the configured to set
	 */
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}

	/**
	 * @return the noise
	 */
	public boolean isNoise() {
		return noise;
	}

	/**
	 * @param noise the noise to set
	 */
	public void setNoise(boolean noise) {
		this.noise = noise;
	}

	/**
	 * @return the photoncount
	 */
	public boolean isPhotoncount() {
		return photoncount;
	}

	/**
	 * @param photoncount the photoncount to set
	 */
	public void setPhotoncount(boolean photoncount) {
		this.photoncount = photoncount;
	}

	/**
	 * @return the photonFluxPerBin
	 */
	public double[] getPhotonFluxPerBin() {
		return photonFluxPerBin;
	}

	/**
	 * @param photonFluxPerBin the photonFluxPerBin to set
	 */
	public void setPhotonFluxPerBin(double[] photonFluxPerBin) {
		this.photonFluxPerBin = photonFluxPerBin;
	}

	/**
	 * @return the channelNames
	 */
	public String[] getChannelNames() {
		return channelNames;
	}

	/**
	 * @param channelNames the channelNames to set
	 */
	public void setChannelNames(String[] channelNames) {
		this.channelNames = channelNames;
	}

	/**
	 * @return the overlap
	 */
	public double getOverlap() {
		return overlap;
	}

	/**
	 * @param overlap the overlap to set
	 */
	public void setOverlap(double overlap) {
		this.overlap = overlap;
	}

}

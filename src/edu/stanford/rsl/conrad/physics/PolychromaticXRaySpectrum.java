/*
 * Copyright (C) 2010-2014 Rotimi X Ojo, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics;

import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * <p> This class creates a model of a polychromatic X Ray spectrum.</p>
 * We model a spectrum as a set of bins.<br>
 *
 * Note that we use the term photon flux here for a quantity that reports the number of photons per area.<br>
 *
 * <p>The default parameters are:<br/>
 * minimum energy = 10keV
 * maximum energy = 150kev
 * resolution delta= 0.5
 * peak voltage = 90kVp
 * time current product = 2.5;</p>
 * 
 * @author Rotimi X Ojo
 * @author Andreas Maier
 */
public class PolychromaticXRaySpectrum {

	//private NumberInterpolatingTreeMap map = new NumberInterpolatingTreeMap();
	private double min = 10.0; // Minimum [keV] 
	private double max = 150.0; // Maximum [keV]
	private double delta = 0.5; // Delta [keV]
	private double peakVoltage = 90.0; //125.0; // Peak Voltage [kVp]			
	private double timeCurrentProduct = 2.5;// 1.0; // Time Current Product [mAs]
	// (X-Ray Tube current 447 mA * Average Pulse Width 5.6 ms )/1000 gives mAs = 2.5
	private double totalPhotonFlux;
	private double[] energies;
	private double[] photonFlux;
	private double avgPhotonEnergy;

	/**
	 * Creates a new polychromatic X-Ray spectrum satisfying default parameters.
	 */
	public PolychromaticXRaySpectrum(){	
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * Creates a new polychromatic X-Ray spectrum with successive energies having a difference of delta
	 * @param delta is the difference between successive energies starting at min
	 */
	public PolychromaticXRaySpectrum(double delta){	
		this.delta = delta;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * Creates a new polychromatic X-Ray spectrum satisfying the parameters below.
	 * @param min is minimum energy in keV
	 * @param max is maximum energy in keV
	 * @param delta is resolution
	 * @param peakVoltage is peak voltage 
	 * @param timeCurrentProduct is time current product
	 */
	public PolychromaticXRaySpectrum(double min, double max, double delta, double peakVoltage, double timeCurrentProduct){
		this.min = min;
		this.max = max;
		this.delta = delta;
		this.peakVoltage = peakVoltage;
		this.timeCurrentProduct = timeCurrentProduct;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * Constructor to create a fully configured spectrum
	 * @param min is minimum energy in keV
	 * @param max is maximum energy in keV
	 * @param delta is resolution
	 * @param peakVoltage is peak voltage 
	 * @param anodeMaterial the target material ("W" or "Mo")
	 * @param timeCurrentProduct the acceleration time times the current in [mAs]
	 * @param mdis amount of air in [m] (i.e. the source detector distance)
	 * @param degrees tube angle in [deg]
	 * @param mmpyrex amount of pyrex filtration in [mm]
	 * @param mmoil amount of oil filtration in [mm]
	 * @param mmlexan amount of lexan filtration in [mm]
	 * @param mmAl amount of Al filtration in [mm]
	 */
	public PolychromaticXRaySpectrum(double min, double max, double delta, double peakVoltage, String anodeMaterial, double timeCurrentProduct, double mdis, double degrees, double mmpyrex, double mmoil, double mmlexan, double mmAl){
		generateSpectrum(min, max, delta, peakVoltage, anodeMaterial, timeCurrentProduct, mdis, degrees, mmpyrex, mmoil, mmlexan, mmAl);
		this.min = min;
		this.max = max;
		this.delta = delta;
		this.peakVoltage = peakVoltage;
		this.timeCurrentProduct = timeCurrentProduct;
	}

	/**
	 * Constructor to create an object of an already known spectrum
	 * @param min minimum energy in keV
	 * @param max maximum energy in keV
	 * @param delta resolution
	 * @param values known values for each bin
	 * @param timeCurrentProduct the acceleration time times the current in [mAs]
	 */
	public PolychromaticXRaySpectrum(double min, double max, double delta,
			double[] values, double timeCurrentProduct) {
		this.min = min;
		this.max = max;
		this.delta = delta;
		this.timeCurrentProduct = timeCurrentProduct;
		int numberOfBins = (int) ((max - min) / delta);
		energies = new double[numberOfBins];
		photonFlux = new double[numberOfBins];
		double weightedIntegral = 0.0;
		double peakVoltage = 0.0;
		for (int i = 0; i < numberOfBins; i++) {
			energies[i] = min + delta * i;
			photonFlux[i] = values[i];
			totalPhotonFlux += photonFlux[i];
			weightedIntegral += energies[i] * photonFlux[i];
			if (values[i] > CONRAD.DOUBLE_EPSILON) {
				peakVoltage = energies[i];
			}
		}
		this.peakVoltage = peakVoltage;
		avgPhotonEnergy = weightedIntegral / totalPhotonFlux;
	}

	/**
	 * Generate a X-Ray spectrum satisfying the parameters below.
	 * @param min is minimum energy in keV
	 * @param max is maximum energy in keV
	 * @param delta is resolution
	 * @param peakVoltage is peak voltage 
	 * @param timeCurrentProduct is time current product
	 */
	private void generateSpectrum(double min, double max, double delta,
			double peakVoltage, String anodeMaterial, double timeCurrentProduct, double mdis, double degrees, double mmpyrex, double mmoil, double mmlexan, double mmAl) {		
		energies = generateEnergies(min, max, delta);
		try {
			photonFlux = XRaySpectrum.generateXRaySpectrum(energies, peakVoltage, anodeMaterial, timeCurrentProduct, mdis, degrees, mmpyrex, mmoil, mmlexan, mmAl);
			totalPhotonFlux = 0;
			double weightedIntegral = 0;
			for (int i = 0; i < energies.length; i++) {
				totalPhotonFlux += photonFlux[i];
				weightedIntegral += energies[i] * photonFlux[i];
			}
			avgPhotonEnergy = weightedIntegral / totalPhotonFlux;
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Photon intensity of spectral bin given at a certain energy (kev) In this
	 * implementation, we round the keV to the next bin of the spectrum.
	 * 
	 * @param energy is energy in [keV]
	 * @return photon intensity in [keV/mm2] 
	 */
	public double getIntensity(double energy) {
		int bin = (int) Math.round((energy - min) / delta);
		if (bin < energies.length)
			return photonFlux[bin] * energies[bin];
		else
			return 0;
	}

	/**
	 * Photon flux of XRay at given energy (kev) In this implementation, we
	 * round the keV to the next bin of the spectrum.
	 * 
	 * @param energy is energy in [keV]
	 * @return x-ray photon flux in [photons/mm2] 
	 */
	public double getPhotonFlux(double energy) {
		int bin = (int) Math.round((energy - min) / delta);
		if (bin < energies.length)
			return photonFlux[bin];
		else
			return 0;
	}

	/**
	 * Returns the total energy of the spectrum per square millimeter, i.e. the integral over all energy bins.
	 * 
	 * @return the total energy in [keV/mm^2]
	 */
	public double getTotalIntensity() {
		return totalPhotonFlux * avgPhotonEnergy;
	}


	/**
	 * Returns the total photon flux of the spectrum, i.e. the integral over all energy bins.
	 * 
	 * @return the total photon flux in [photons/mm^2]
	 */
	public double getTotalPhotonFlux() {
		return totalPhotonFlux;
	}

	/**
	 * Computes the average phonton energy of this spectrum given as the weighted sum of energies multiplied with photon flux normalized by the photon flux.
	 * 
	 * @return average photon enegy in [keV]
	 */
	public double getAveragePhotonEnergy() {
		return avgPhotonEnergy;
	}


	/**
	 * @return photon energies in kev
	 */
	public double[] getPhotonEnergies() {
		return energies;
	}
	
	/**
	 * set photon energies in kev
	 */
	public void setPhotonEnergies(double[] e) {
		energies=e;
	}

	/**
	 * Determine the number of discrete energies used to describe the spectrum
	 * @return 0 if there are energies in the spectrum.
	 */
	public int size() {
		return energies.length;
	}

	/**
	 * returns the photon flux array containing for a spectral bin.
	 * @return the photon flux the array of fluxes in [photons/mm^2]
	 */
	public double[] getPhotonFlux() {
		return photonFlux;
	}
	
	/**
	 *
	 * @return the timeCurrentProduct of the spectrum
	 */
	public double getTimeCurrentProduct(){
		return timeCurrentProduct;
	}

	/**
	 * @param photonFlux the photon flux to set
	 */
	public void setPhotonFlux(double[] photonFlux) {
		this.photonFlux = photonFlux;
	}

	private double[] generateEnergies(double min, double max, double delta) {
		int steps = (int) ((max - min) / delta);
		double[] pen = new double[steps];
		for (int i = 0; i < steps; i++) {
			pen[i] = min + (i * delta);
		}
		return pen;
	}

	/**
	 * @return the min
	 */
	public double getMin() {
		return min;
	}

	/**
	 * @param min the min to set
	 */
	public void setMin(double min) {
		this.min = min;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * @return the max
	 */
	public double getMax() {
		return max;
	}

	/**
	 * @param max the max to set
	 */
	public void setMax(double max) {
		this.max = max;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * @return the delta
	 */
	public double getDelta() {
		return delta;
	}

	/**
	 * @param delta the delta to set
	 */
	public void setDelta(double delta) {
		this.delta = delta;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * @return the peakVoltage
	 */
	public double getPeakVoltage() {
		return peakVoltage;
	}

	/**
	 * @param peakVoltage the peakVoltage to set
	 */
	public void setPeakVoltage(double peakVoltage) {
		this.peakVoltage = peakVoltage;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

	/**
	 * @param timeCurrentProduct the timeCurrentProduct to set
	 */
	public void setTimeCurrentProduct(double timeCurrentProduct) {
		this.timeCurrentProduct = timeCurrentProduct;
		generateSpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, 1, 10, 2.38, 3.06, 2.66, 10.5);
	}

}

/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.absorption;

import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.physics.materials.utils.MaterialUtils;
import edu.stanford.rsl.conrad.utils.StatisticsUtil;


/**
 * <p>Polychromatic Absorption Model with dynamic spectrum support. <br> This class models the absorption of an X-Ray spectrum by well-defined materials. </p>
 * 
 * @author Rotimi X Ojo, Andreas Maier
 */
public class PolychromaticAbsorptionModel extends AbsorptionModel {	

	/**
	 * 
	 */
	private static final long serialVersionUID = -3430401648474324638L;
	private AttenuationType att = AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION;

	protected PolychromaticXRaySpectrum inputSpectrum;

	protected double [] energies;
	protected double [] photonFlux;
	protected HashMap<Material, double []> attenuationCoefficientsMap;
	



	/**
	 * Compute the absorption of a given X-Ray spectrum along a segmented path.
	 * @param segments are material dependent segmentation of the path followed by an arbitrary X-Ray
	 * @return the log attenuation of line integral of the product of X-Ray Intensities over a range of energies and energy dependent attenuation.
	 */
	@Override
	public double evaluateLineIntegral(ArrayList<PhysicalObject> segments) {

		double intensity = computeIntensity(segments, energies[0], energies[energies.length-1], false, true);
		// we do not require the pixel area here, as we normalize with the total spectral intensity.
		double value = intensity/inputSpectrum.getTotalIntensity();
		
		//Due to the accumulation round off error, it is possible for output intensity to be marginally greater  
		//than input intensity in vacuum. We deal with this problem by bounding output intensity by the input 
		//intensity.
		if(value > 1){
			value = 1;
			//throw new RuntimeException("PolychromaticAbsorptionModel: numerical instability found.");
		}
		//End

		return  -Math.log(value);
	}

	private int convertToIndex(double keV){
		return (int)Math.round((keV-energies[0])/(energies[1]-energies[0]));
	}

	/**
	 * Change the X-Ray spectrum used by the absorption model. This method is important for implementing Dynamic Spectrum and XML Serialization.
	 * @param spectrum is the new spectrum of the absorption model.
	 */
	public void setInputSpectrum(PolychromaticXRaySpectrum  spectrum){
		this.inputSpectrum = spectrum;
		// precompute energies:
		energies = inputSpectrum.getPhotonEnergies();
		// precompute intensities:
		photonFlux = new double[energies.length];
		for (int i = 0; i < energies.length; i++){
			photonFlux[i]=inputSpectrum.getPhotonFlux(energies[i]);
		}
		// precompute absorption spectra for all known materials in database:
		attenuationCoefficientsMap = MaterialUtils.loadAttenuationCoefficients(energies, att);
	}

	/**
	 * Retrieve the current input X-Ray spectrum used by the absorption model
	 * @return the current X-Ray spectrum used by the absorption model
	 */
	public PolychromaticXRaySpectrum getInputSpectrum(){
		return inputSpectrum;
	}

	@Override
	public String toString() {
		return "Polychromatic Absorption Model";
	}

	public double getMaximalEnergy(){
		return energies[energies.length-1];
	}

	public double getMinimalEnergy(){
		return energies[0];
	}

	@Override
	public void configure() throws Exception {
		inputSpectrum = new PolychromaticXRaySpectrum(2.5);
		// we use this method to preload different material properties in order to be able to execute the code in parallel efficiently.
		setInputSpectrum(inputSpectrum);
	}

	@Override
	public boolean isConfigured() {
		return true;
	}


	public double getPhotonFluxIntegral(double startEnergy, double stopEnergy) {
		int start = convertToIndex(startEnergy);
		int end = convertToIndex(stopEnergy);
		double sum = 0;
		if (start < 0) start =0;
		if (end >= photonFlux.length) end = photonFlux.length-1;
		for (int i= start; i<=end; i++){
			sum+=this.photonFlux[i];
		}
		return sum;
	}

	/**
	 * Computes the integral over the spectrum given a start and an end energy.
	 * If energyIntegrating is true the accumulated energy is returned otherwise the photon count.
	 * @param segments the path segments
	 * @param startEnergy the start energy [keV]
	 * @param endEnergy the end energy [keV]
	 * @param noise if true noise is generated
	 * @param energyIntegrating if true the photon count is weighted with the energy
	 * @return the intensity.
	 */
	public double computeIntensity(ArrayList<PhysicalObject> segments,
			double startEnergy, double endEnergy, boolean noise, boolean energyIntegrating) {
		double intensity = 0;
		int start = convertToIndex(startEnergy);
		int end = convertToIndex(endEnergy);
		if (start < 0) start =0;
		if (end >= photonFlux.length) end = photonFlux.length-1;
		double lens [] = new double[segments.size()];
		for (int j = 0; j < segments.size(); j++){
			lens[j] = ((Edge)segments.get(j).getShape()).getLength();
		}
		for (int e = start; e <= end; e++){
			double sum = 0;
			for (int j = 0; j < segments.size(); j++){
				PhysicalObject o = segments.get(j);
				double att = getAttenuationCoefficients(o.getMaterial())[e];
				sum +=  att * lens[j];
			}
			// TODO: Normalization is never considered in the backprojectors, 
						// 		 thus, iteratively applying forward and backward projections
						//		 would yield to a scaling issue!
						//
						// length is in [mm]
						// attenuation is in [g/cm^3]
						// conversion from [g*mm/mc^3] = [g*0.1cm/cm^3] to [g/cm^2]
						// --> sum/10.0;
			double afterAttenuation = photonFlux[e] * Math.exp(-sum/10);

			if (noise) {
				double photonsWithNoise = StatisticsUtil.poissonRandomNumber(afterAttenuation);
				if (energyIntegrating){
					photonsWithNoise *= energies[e];
				}
				intensity += photonsWithNoise;
			} else {
				if (energyIntegrating){
					afterAttenuation *= energies[e];
				}
				intensity += afterAttenuation;
			}
		}
		return intensity;
	}

	/**
	 * 
	 * @return the total intensity of the input spectrum
	 */
	public double getTotalIntensity() {
		return inputSpectrum.getTotalIntensity();
	}
	
	/**
	 * 
	 * @return the total photon flux of the input spectrum
	 */
	public double getTotalPhotonFlux() {
		return inputSpectrum.getTotalPhotonFlux();
	}
	
	/**
	 * 
	 * @return the time current product of the input spectrum
	 */
	public double getTimeCurrentProduct(){
		return inputSpectrum.getTimeCurrentProduct();
	}
	
	/**
	 * @return the att
	 */
	public AttenuationType getAtt() {
		return att;
	}


	/**
	 * @param att the att to set
	 */
	public void setAtt(AttenuationType att) {
		this.att = att;
	}



	public double [] getAttenuationCoefficients(Material mat){
		double [] result = attenuationCoefficientsMap.get(mat);
		if (result == null){
			MaterialUtils.updateAttenuationCoefficientsMap(attenuationCoefficientsMap, energies, mat, att);
			result = attenuationCoefficientsMap.get(mat);
		}
		return result;
	}

}

package edu.stanford.rsl.conrad.physics;

import java.util.Arrays;

import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.utils.BilinearInterpolatingDoubleArray;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.LinearInterpolatingDoubleArray;

/**
 * Class to lookup certain energy dependent mass attenuation coefficients.
 * 
 * All values are based on the <a href="http://physics.nist.gov/PhysRefData/XrayMassCoef/>NIST X-Ray Mass Attenuation Coefficients</a>.
 * 
 * @author akmaier
 *
 */

@Deprecated
public class EnergyDependentCoefficients {

	/**
	 * Supported materials
	 */
	@Deprecated
	public static enum Material {Water, Aluminum, CorticalBone, SoftBone, SoftTissue, Titanium, Gold, Teflon, Polysterene, Lucite, PDS2BaseMaterial, QrmHA1100Material, PlasticineMaterial, LegCylinder, CsI};

	/**
	 * Returns a LinearInterpolatingDoubleArray which reports the photon mass attenuation coefficients over rho [cm^2/g] for the respective energies [MeV].
	 * 
	 * @param material the material to look up
	 * @return the LUT
	 * @throws Exception
	 */
	public static LinearInterpolatingDoubleArray getPhotonMassAttenuationOverRhoLUT (Material material) throws Exception{
		switch (material) { 
		case Water:
			return new LinearInterpolatingDoubleArray(waterEnergies, waterMuOverRho);
		case Aluminum:
			return new LinearInterpolatingDoubleArray(aluminumEnergies, aluminumMuOverRho);
		case CorticalBone:
			return new LinearInterpolatingDoubleArray(corticalBoneEnergies, corticalBoneMuOverRho);
		case SoftBone:
			return new LinearInterpolatingDoubleArray(softBoneEnergies, softBoneMuOverRho);
		case SoftTissue:
			return new LinearInterpolatingDoubleArray(softTissueEnergies, softTissueMuOverRho);
		case Titanium:
			return new LinearInterpolatingDoubleArray(titaniumEnergies, titaniumMuOverRho);
		case Gold:
			return new LinearInterpolatingDoubleArray(goldEnergies, goldMuOverRho);
		case Teflon:
			return new LinearInterpolatingDoubleArray(teflonEnergies, teflonMuOverRho);
		case CsI:
			return new LinearInterpolatingDoubleArray(CsIEnergies, CsIMuOverRho);
		default:
			throw new Exception ("Unknown Material");
		}
	}
	
	/**
	 * Returns the mass density of the material in [g/cm^3]
	 * 
	 * @param material the material to look up
	 * @return the mass density
	 */
	public static double getMassDensity (Material material){
		switch (material) { 
		case Water:
			return Constants.Water_mdensity;
		case Aluminum:
			return Constants.Al_mdensity;
		case CorticalBone:
			return Constants.CorticalBone_mdensity;
		case SoftBone:
			return Constants.SoftBone_mdensity;
		case SoftTissue:
			return Constants.SoftTissue_mdensity;
		case Titanium:
			return Constants.Ti_mdensity;
		case Gold:
			return Constants.Au_mdensity;
		case Teflon:
			return Constants.TEFLON_mdensity;
		case Lucite:
			return Constants.Lucite_mdensity;
		case Polysterene:
			return Constants.Polystyrene_mdensity;
		case CsI:
			return Constants.CsI_mdensity;
		default:
			throw new RuntimeException ("Unknown Material");
		}
	}
	
	/**
	 * Returns the CT number of the material in [g/cm^3]
	 * 
	 * @param material the material to look up
	 * @return the CT number
	 */
	public static double getCTNumber (Material material){
		switch (material) { 
		case Water:
			return Constants.WaterCTValue;
		case Teflon:
			return Constants.TeflonCTValue;
		case SoftBone:
			return Constants.SoftBoneCTValue;
		case SoftTissue:
			return Constants.SoftTissueCTValue;
		case Polysterene:
			return Constants.PolystyreneCTValue;
		case Lucite:
			return Constants.LuciteCTNumber;
		case PDS2BaseMaterial:
			return Constants.PDS2BaseMaterialCTNumber;
		case PlasticineMaterial:
			return Constants.PlasticineMaterialCTNumber;
		case QrmHA1100Material:
			return Constants.QrmHA1100MaterialCTNumber;
		case LegCylinder:
			return Constants.LegCylinder;
		default:
			throw new RuntimeException ("Unknown Material");
		}
	}
	
	/**
	 * Returns a LinearInterpolatingDoubleArray which reports the photon mass attenuation coeffients [1/cm] for the respective energies [MeV].
	 * 
	 * @param material the material to look up
	 * @return the LUT
	*/
	public static LinearInterpolatingDoubleArray getPhotonMassAttenuationLUT (Material material) {
		switch (material) { 
		case Water:
			return new LinearInterpolatingDoubleArray(waterEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(waterMuOverRho, waterMuOverRho.length), Constants.Water_mdensity));
		case Aluminum:
			return new LinearInterpolatingDoubleArray(aluminumEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(aluminumMuOverRho, aluminumMuOverRho.length), Constants.Al_mdensity));
		case CorticalBone:
			return new LinearInterpolatingDoubleArray(corticalBoneEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(corticalBoneMuOverRho, corticalBoneMuOverRho.length), Constants.CorticalBone_mdensity));
		case SoftBone:
			return new LinearInterpolatingDoubleArray(softBoneEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(softBoneMuOverRho, softBoneMuOverRho.length), Constants.SoftBone_mdensity));
		case SoftTissue:
			return new LinearInterpolatingDoubleArray(softTissueEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(softTissueMuOverRho, softTissueMuOverRho.length), Constants.SoftTissue_mdensity));
		case Titanium:
			return new LinearInterpolatingDoubleArray(titaniumEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(titaniumMuOverRho, titaniumMuOverRho.length), Constants.Ti_mdensity));
		case Gold:
			return new LinearInterpolatingDoubleArray(goldEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(goldMuOverRho, goldMuOverRho.length), Constants.Au_mdensity));
		case Teflon:
			return new LinearInterpolatingDoubleArray(teflonEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(teflonMuOverRho, teflonMuOverRho.length), Constants.TEFLON_mdensity));
		case CsI:
			return new LinearInterpolatingDoubleArray(CsIEnergies, DoubleArrayUtil.multiply(Arrays.copyOf(CsIMuOverRho, CsIMuOverRho.length), Constants.CsI_mdensity));
		default:
			throw new RuntimeException ("Unknown Material");
		}
	}

	/**
	 * Returns a LinearInterpolatingDoubleArray which reports the mass-energy absorption coeffients [cm^2/g] for the respective energies [MeV].
	 * @param material the material to look up
	 * @return the LUT
	 * @throws Exception
	 */
	public static LinearInterpolatingDoubleArray getMassEnergyAbsorptionLUT (Material material) throws Exception{
		switch (material) { 
		case Water:
			return new LinearInterpolatingDoubleArray(waterEnergies, waterMuEnOverRho);
		case Aluminum:
			return new LinearInterpolatingDoubleArray(aluminumEnergies, aluminumMuEnOverRho);
		case CorticalBone:
			return new LinearInterpolatingDoubleArray(corticalBoneEnergies, corticalBoneMuEnOverRho);
		case SoftBone:
			return new LinearInterpolatingDoubleArray(softBoneEnergies, softBoneMuEnOverRho);
		case SoftTissue:
			return new LinearInterpolatingDoubleArray(softTissueEnergies, softTissueMuEnOverRho);
		case Titanium:
			return new LinearInterpolatingDoubleArray(titaniumEnergies, titaniumMuEnOverRho);
		case Gold:
			return new LinearInterpolatingDoubleArray(goldEnergies, goldMuEnOverRho);
		case Teflon:
			return new LinearInterpolatingDoubleArray(teflonEnergies, teflonMuEnOverRho);
		case CsI:
			return new LinearInterpolatingDoubleArray(CsIEnergies, CsIMuEnOverRho);
		default:
			throw new Exception ("Unknown Material");
		}
	}
	
	/**
	 * Generates a BilinearInterpolatingDoubleArray which can be used to look up lambda values.
	 * @param maxWater the maximal value of water corrected observations
	 * @param maxPassedMaterial the maximal value of total passed hard material attenuation
	 * @param stepSize the sampling step-size
	 * @param energies the x-ray energies
	 * @param xRaySpectrum the sampled spectrum
	 * @param softMaterial the photon mass attenuation coefficients of the soft material
	 * @param hardMaterial the photon mass attenuation coefficients of the hard material
	 * @return the beam hardening lookup table
	 */
	public static BilinearInterpolatingDoubleArray getBeamHardeningLookupTable(double maxWater, double maxPassedMaterial, double stepSize, double [] energies, double [] xRaySpectrum, LinearInterpolatingDoubleArray softMaterial, LinearInterpolatingDoubleArray hardMaterial){
		int indicesWater = (int) Math.ceil(maxWater / stepSize);
		int indicesPassedMaterial = (int) Math.ceil(maxPassedMaterial / stepSize);
		double [] waterCorrectedValues = new double[indicesWater];
		double [] passedMaterial = new double [indicesPassedMaterial];
		for (int i = 0; i< waterCorrectedValues.length; i++){
			waterCorrectedValues[i] = i * stepSize;
		}
		for (int i = 0; i< passedMaterial.length; i++){
			passedMaterial[i] = i * stepSize;
		}
		return getBeamHardeningLookupTable(waterCorrectedValues, passedMaterial, energies, xRaySpectrum, softMaterial, hardMaterial);
	}
	
	/**
	 * Generates a BilinearInterpolatingDoubleArray which can be used to look up lambda values.
	 * @param waterCorrectedValues the water corrected values sampling
	 * @param passedMaterial the passed material sampling
	 * @param energies the x-ray energies
	 * @param xRaySpectrum the sampled spectrum
	 * @param softMaterialAttenuationCoefficients the photon mass attenuation coefficients of the soft material
	 * @param hardMaterialAttenuationCoefficients the photon mass attenuation coefficients of the hard material
	 * @return the beam hardening lookup table
	 */
	public static BilinearInterpolatingDoubleArray getBeamHardeningLookupTable(double [] waterCorrectedValues, double [] passedMaterial, double[] energies, double [] xRaySpectrum, LinearInterpolatingDoubleArray softMaterialAttenuationCoefficients, LinearInterpolatingDoubleArray hardMaterialAttenuationCoefficients){
		double [][] lutValues = null;
		try {
			double [] soft = new double [energies.length];
			double [] hard = new double [energies.length];
			for (int i = 0; i < energies.length; i++){
				soft[i] = softMaterialAttenuationCoefficients.getValue(energies[i]/1000.0);
				hard[i] = hardMaterialAttenuationCoefficients.getValue(energies[i]/1000.0);
			}
			LambdaFunction [] lambdas = new LambdaFunction[waterCorrectedValues.length*passedMaterial.length];
			double factor = 1.0;
			for (int i = 0; i<waterCorrectedValues.length; i++){
				double epsilon = 0;
				for (int k = 0; k< xRaySpectrum.length; k++){
					epsilon += xRaySpectrum[k] * Math.exp(-soft[k]*waterCorrectedValues[i]*factor);
				}
				for (int j = 0; j<passedMaterial.length; j++){
					//System.out.println(waterCorrectedValues[i] + " " + passedMaterial[j]);
					lambdas[(j*waterCorrectedValues.length)+i] = new LambdaFunction(xRaySpectrum, soft, hard, waterCorrectedValues[i]*factor, passedMaterial[j]*factor, epsilon);
				}
			}
			ParallelThreadExecutor exec = new ParallelThreadExecutor(lambdas);
			exec.execute();		
			lutValues = new double[waterCorrectedValues.length][passedMaterial.length];
			for (int i = 0; i<waterCorrectedValues.length; i++){
				for (int j = 0; j<passedMaterial.length; j++){		
					lutValues[i][j] = lambdas[(j*waterCorrectedValues.length)+i].getOptimalLambda();
					//System.out.println(lutValues[i][j] + " " + lambdas[(j*waterCorrectedValues.length)+i].evaluate(lutValues[i][j])) ;
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return new BilinearInterpolatingDoubleArray(waterCorrectedValues, passedMaterial, lutValues);
	}

	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] waterEnergies = {1.00E-03, 1.50E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] waterMuOverRho = {4.08E+03, 1.38E+03, 6.17E+02, 1.93E+02, 8.28E+01, 4.26E+01, 2.46E+01, 1.04E+01, 5.33E+00, 1.67E+00, 8.10E-01, 3.76E-01, 2.68E-01, 2.27E-01, 2.06E-01, 1.84E-01, 1.71E-01, 1.51E-01, 1.37E-01, 1.19E-01, 1.06E-01, 9.69E-02, 8.96E-02, 7.87E-02, 7.07E-02, 6.32E-02, 5.75E-02, 4.94E-02, 3.97E-02, 3.40E-02, 3.03E-02, 2.77E-02, 2.43E-02, 2.22E-02, 1.94E-02, 1.81E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] waterMuEnOverRho = {4.07E+03, 1.37E+03, 6.15E+02, 1.92E+02, 8.19E+01, 4.19E+01, 2.41E+01, 9.92E+00, 4.94E+00, 1.37E+00, 5.50E-01, 1.56E-01, 6.95E-02, 4.22E-02, 3.19E-02, 2.60E-02, 2.55E-02, 2.76E-02, 2.97E-02, 3.19E-02, 3.28E-02, 3.30E-02, 3.28E-02, 3.21E-02, 3.10E-02, 2.97E-02, 2.83E-02, 2.61E-02, 2.28E-02, 2.07E-02, 1.92E-02, 1.81E-02, 1.66E-02, 1.57E-02, 1.44E-02, 1.38E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] corticalBoneEnergies = {1.00E-03, 1.04E-03, 1.07E-03, 1.07E-03, 1.18E-03, 1.31E-03, 1.31E-03, 1.50E-03, 2.00E-03, 2.15E-03, 2.15E-03, 2.30E-03, 2.47E-03, 2.47E-03, 3.00E-03, 4.00E-03, 4.04E-03, 4.04E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] corticalBoneMuOverRho = {3.78E+03, 3.45E+03, 3.15E+03, 3.16E+03, 2.43E+03, 1.87E+03, 1.88E+03, 1.30E+03, 5.87E+02, 4.82E+02, 7.11E+02, 5.92E+02, 4.91E+02, 4.96E+02, 2.96E+02, 1.33E+02, 1.30E+02, 3.33E+02, 1.92E+02, 1.17E+02, 5.32E+01, 2.85E+01, 9.03E+00, 4.00E+00, 1.33E+00, 6.66E-01, 4.24E-01, 3.15E-01, 2.23E-01, 1.86E-01, 1.48E-01, 1.31E-01, 1.11E-01, 9.91E-02, 9.02E-02, 8.33E-02, 7.31E-02, 6.57E-02, 5.87E-02, 5.35E-02, 4.61E-02, 3.75E-02, 3.26E-02, 2.95E-02, 2.73E-02, 2.47E-02, 2.31E-02, 2.13E-02, 2.07E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] corticalBoneMuEnOverRho = {3.77E+03, 3.44E+03, 3.14E+03, 3.15E+03, 2.43E+03, 1.87E+03, 1.88E+03, 1.29E+03, 5.85E+02, 4.80E+02, 6.96E+02, 5.79E+02, 4.81E+02, 4.86E+02, 2.90E+02, 1.30E+02, 1.27E+02, 3.01E+02, 1.76E+02, 1.09E+02, 4.99E+01, 2.68E+01, 8.39E+00, 3.60E+00, 1.07E+00, 4.51E-01, 2.34E-01, 1.40E-01, 6.90E-02, 4.59E-02, 3.18E-02, 3.00E-02, 3.03E-02, 3.07E-02, 3.07E-02, 3.05E-02, 2.97E-02, 2.88E-02, 2.75E-02, 2.62E-02, 2.42E-02, 2.15E-02, 1.98E-02, 1.86E-02, 1.79E-02, 1.70E-02, 1.64E-02, 1.59E-02, 1.57E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] softBoneEnergies = {0.00E+00, 1.50E-03, 2.00E-03, 3.00E-03, 4.00E-03, 4.04E-03, 4.04E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] softBoneMuOverRho = {3.21E+03, 1.08E+03, 4.88E+02, 1.54E+02, 6.72E+01, 6.54E+01, 2.25E+02, 1.30E+02, 7.94E+01, 3.62E+01, 1.94E+01, 6.22E+00, 2.80E+00, 9.75E-01, 5.18E-01, 3.51E-01, 2.74E-01, 2.08E-01, 1.79E-01, 1.48E-01, 1.33E-01, 1.14E-01, 1.01E-01, 9.23E-02, 8.53E-02, 7.48E-02, 6.72E-02, 6.01E-02, 5.47E-02, 4.71E-02, 3.80E-02, 3.28E-02, 2.94E-02, 2.70E-02, 2.40E-02, 2.22E-02, 1.99E-02, 1.89E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] softBoneMuEnOverRho = {3.20E+03, 1.08E+03, 4.86E+02, 1.53E+02, 6.62E+01, 6.44E+01, 2.01E+02, 1.18E+02, 7.29E+01, 3.36E+01, 1.81E+01, 5.69E+00, 2.45E+00, 7.34E-01, 3.12E-01, 1.65E-01, 1.01E-01, 5.37E-02, 3.86E-02, 3.03E-02, 2.98E-02, 3.08E-02, 3.14E-02, 3.15E-02, 3.13E-02, 3.05E-02, 2.95E-02, 2.82E-02, 2.69E-02, 2.48E-02, 2.18E-02, 1.99E-02, 1.86E-02, 1.77E-02, 1.64E-02, 1.57E-02, 1.48E-02, 1.44E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] aluminumEnergies = {1.00E-03, 1.50E-03, 1.56E-03, 1.56E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	//30-AUG-82 HOWERTON, MOD. BY JRC : public static double [] aluminumEnergies = {0.001, 0.00125, 0.0015, 0.001559, 0.00156, 0.00175, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1, 1.022, 1.25, 1.5, 1.75, 2, 2.044, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] aluminumMuOverRho = {1.19E+03, 4.02E+02, 3.62E+02, 3.96E+03, 2.26E+03, 7.88E+02, 3.61E+02, 1.93E+02, 1.15E+02, 5.03E+01, 2.62E+01, 7.96E+00, 3.44E+00, 1.13E+00, 5.69E-01, 3.68E-01, 2.78E-01, 2.02E-01, 1.70E-01, 1.38E-01, 1.22E-01, 1.04E-01, 9.28E-02, 8.45E-02, 7.80E-02, 6.84E-02, 6.15E-02, 5.50E-02, 5.01E-02, 4.32E-02, 3.54E-02, 3.11E-02, 2.84E-02, 2.66E-02, 2.44E-02, 2.32E-02, 2.20E-02, 2.17E-02};
	//30-AUG-82 HOWERTON, MOD. BY JRC : public static double [] aluminumMuOverRho = {1177, 644.2, 394, 356.2, 4241, 3199, 2305, 1293, 806.9, 530.8, 369.4, 265.7, 197.9, 150.8, 117.6, 93.16, 75.09, 50.95, 26.35, 13.5, 7.873, 4.98, 3.375, 1.789, 1.101, 0.7516, 0.5571, 0.4387, 0.3632, 0.3117, 0.2757, 0.2489, 0.2289, 0.2015, 0.1706, 0.1499, 0.1381, 0.1291, 0.1225, 0.1118, 0.1043, 0.09789, 0.09279, 0.08828, 0.08447, 0.08102, 0.07801, 0.0752, 0.0727, 0.06841, 0.06145, 0.0608, 0.05493, 0.05007, 0.04629, 0.04325, 0.04276, 0.03865, 0.03542, 0.03297, 0.03109, 0.02962, 0.02839, 0.0273, 0.0264, 0.02567, 0.02506, 0.02415, 0.02318, 0.02217, 0.02172, 0.02165, 0.02168, 0.02176, 0.02196, 0.02214, 0.0224, 0.02278, 0.02307, 0.02325, 0.02344, 0.02364, 0.02385, 0.02428, 0.02513};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] aluminumMuEnOverRho = {1.18E+03, 4.00E+02, 3.60E+02, 3.83E+03, 2.20E+03, 7.73E+02, 3.55E+02, 1.90E+02, 1.13E+02, 4.92E+01, 2.54E+01, 7.49E+00, 3.09E+00, 8.78E-01, 3.60E-01, 1.84E-01, 1.10E-01, 5.51E-02, 3.79E-02, 2.83E-02, 2.75E-02, 2.82E-02, 2.86E-02, 2.87E-02, 2.85E-02, 2.78E-02, 2.69E-02, 2.57E-02, 2.45E-02, 2.27E-02, 2.02E-02, 1.88E-02, 1.80E-02, 1.74E-02, 1.68E-02, 1.65E-02, 1.63E-02, 1.63E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] softTissueEnergies = {1.00E-03, 1.04E-03, 1.07E-03, 1.07E-03, 1.50E-03, 2.00E-03, 2.15E-03, 2.15E-03, 2.30E-03, 2.47E-03, 2.47E-03, 2.64E-03, 2.82E-03, 2.82E-03, 3.00E-03, 3.61E-03, 3.61E-03, 4.00E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] softTissueMuOverRho = {3.71E+03, 3.39E+03, 3.09E+03, 3.10E+03, 1.25E+03, 5.60E+02, 4.58E+02, 4.65E+02, 3.80E+02, 3.10E+02, 3.16E+02, 2.61E+02, 2.16E+02, 2.19E+02, 1.84E+02, 1.07E+02, 1.11E+02, 8.16E+01, 4.22E+01, 2.46E+01, 1.04E+01, 5.38E+00, 1.70E+00, 8.23E-01, 3.79E-01, 2.69E-01, 2.26E-01, 2.05E-01, 1.82E-01, 1.69E-01, 1.49E-01, 1.36E-01, 1.18E-01, 1.05E-01, 9.60E-02, 8.87E-02, 7.79E-02, 7.01E-02, 6.27E-02, 5.70E-02, 4.90E-02, 3.93E-02, 3.37E-02, 3.00E-02, 2.74E-02, 2.40E-02, 2.19E-02, 1.92E-02, 1.79E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] softTissueMuEnOverRho = {3.70E+03, 3.38E+03, 3.08E+03, 3.09E+03, 1.25E+03, 5.58E+02, 4.57E+02, 4.63E+02, 3.78E+02, 3.09E+02, 3.14E+02, 2.59E+02, 2.14E+02, 2.17E+02, 1.82E+02, 1.06E+02, 1.09E+02, 8.03E+01, 4.14E+01, 2.39E+01, 9.94E+00, 4.99E+00, 1.40E+00, 5.66E-01, 1.62E-01, 7.22E-02, 4.36E-02, 3.26E-02, 2.62E-02, 2.55E-02, 2.75E-02, 2.94E-02, 3.16E-02, 3.25E-02, 3.27E-02, 3.25E-02, 3.18E-02, 3.07E-02, 2.94E-02, 2.81E-02, 2.58E-02, 2.26E-02, 2.05E-02, 1.90E-02, 1.79E-02, 1.64E-02, 1.55E-02, 1.42E-02, 1.36E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] goldEnergies = {1.00E-03, 1.50E-03, 2.00E-03, 2.21E-03, 2.21E-03, 2.25E-03, 2.29E-03, 2.29E-03, 2.51E-03, 2.74E-03, 2.74E-03, 3.00E-03, 3.15E-03, 3.15E-03, 3.28E-03, 3.42E-03, 3.42E-03, 4.00E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.19E-02, 1.19E-02, 1.28E-02, 1.37E-02, 1.37E-02, 1.40E-02, 1.44E-02, 1.44E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 8.07E-02, 8.07E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] goldMuOverRho = {4.65E+03, 2.09E+03, 1.14E+03, 9.19E+02, 9.97E+02, 1.39E+03, 2.26E+03, 2.39E+03, 2.38E+03, 2.20E+03, 2.54E+03, 2.05E+03, 1.82E+03, 1.93E+03, 1.75E+03, 1.59E+03, 1.65E+03, 1.14E+03, 6.66E+02, 4.25E+02, 2.07E+02, 1.18E+02, 7.58E+01, 1.87E+02, 1.55E+02, 1.28E+02, 1.76E+02, 1.77E+02, 1.59E+02, 1.83E+02, 1.64E+02, 7.88E+01, 2.75E+01, 1.30E+01, 7.26E+00, 4.53E+00, 2.19E+00, 2.14E+00, 8.90E+00, 5.16E+00, 1.86E+00, 9.21E-01, 3.74E-01, 2.18E-01, 1.53E-01, 1.19E-01, 8.60E-02, 6.95E-02, 5.79E-02, 5.17E-02, 4.57E-02, 4.20E-02, 4.17E-02, 4.24E-02, 4.36E-02, 4.63E-02, 4.93E-02, 5.60E-02, 6.14E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] goldMuEnOverRho = {4.64E+03, 2.08E+03, 1.13E+03, 9.07E+02, 9.84E+02, 1.36E+03, 2.21E+03, 2.34E+03, 2.33E+03, 2.15E+03, 2.48E+03, 2.01E+03, 1.78E+03, 1.89E+03, 1.71E+03, 1.55E+03, 1.62E+03, 1.12E+03, 6.51E+02, 4.14E+02, 2.00E+02, 1.13E+02, 7.13E+01, 1.52E+02, 1.27E+02, 1.07E+02, 1.38E+02, 1.32E+02, 1.25E+02, 1.43E+02, 1.29E+02, 6.52E+01, 2.35E+01, 1.11E+01, 6.12E+00, 3.75E+00, 1.72E+00, 1.68E+00, 2.51E+00, 2.07E+00, 1.03E+00, 5.56E-01, 2.29E-01, 1.27E-01, 8.52E-02, 6.41E-02, 4.43E-02, 3.53E-02, 2.92E-02, 2.59E-02, 2.33E-02, 2.30E-02, 2.43E-02, 2.58E-02, 2.73E-02, 2.97E-02, 3.16E-02, 3.45E-02, 3.57E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] titaniumEnergies = {1.00E-03, 1.50E-03, 2.00E-03, 3.00E-03, 4.00E-03, 4.97E-03, 4.97E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] titaniumMuOverRho = {5.87E+03, 2.10E+03, 9.86E+02, 3.32E+02, 1.52E+02, 8.38E+01, 6.88E+02, 6.84E+02, 4.32E+02, 2.02E+02, 1.11E+02, 3.59E+01, 1.59E+01, 4.97E+00, 2.21E+00, 1.21E+00, 7.66E-01, 4.05E-01, 2.72E-01, 1.65E-01, 1.31E-01, 1.04E-01, 9.08E-02, 8.19E-02, 7.53E-02, 6.57E-02, 5.89E-02, 5.26E-02, 4.80E-02, 4.18E-02, 3.51E-02, 3.17E-02, 2.98E-02, 2.87E-02, 2.76E-02, 2.73E-02, 2.76E-02, 2.84E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] titaniumMuEnOverRho = {5.86E+03, 2.09E+03, 9.82E+02, 3.30E+02, 1.49E+02, 8.19E+01, 5.68E+02, 5.66E+02, 3.69E+02, 1.79E+02, 1.00E+02, 3.31E+01, 1.47E+01, 4.49E+00, 1.90E+00, 9.74E-01, 5.63E-01, 2.42E-01, 1.31E-01, 5.39E-02, 3.73E-02, 3.01E-02, 2.86E-02, 2.80E-02, 2.76E-02, 2.66E-02, 2.56E-02, 2.44E-02, 2.33E-02, 2.17E-02, 1.99E-02, 1.91E-02, 1.88E-02, 1.88E-02, 1.90E-02, 1.93E-02, 2.01E-02, 2.07E-02};
	/**
	 * Energies [Mev] at which the coefficients were meassured. 
	 */
	public static double [] teflonEnergies = {1.00E-03, 1.50E-03, 2.00E-03, 3.00E-03, 4.00E-03, 5.00E-03, 6.00E-03, 8.00E-03, 1.00E-02, 1.50E-02, 2.00E-02, 3.00E-02, 4.00E-02, 5.00E-02, 6.00E-02, 8.00E-02, 1.00E-01, 1.50E-01, 2.00E-01, 3.00E-01, 4.00E-01, 5.00E-01, 6.00E-01, 8.00E-01, 1.00E+00, 1.25E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00, 5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01, 2.00E+01};
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] teflonMuOverRho = {4.82E+03, 1.67E+03, 7.60E+02, 2.41E+02, 1.05E+02, 5.41E+01, 3.14E+01, 1.33E+01, 6.81E+00, 2.09E+00, 9.67E-01, 4.03E-01, 2.65E-01, 2.13E-01, 1.88E-01, 1.63E-01, 1.50E-01, 1.31E-01, 1.19E-01, 1.03E-01, 9.19E-02, 8.38E-02, 7.75E-02, 6.80E-02, 6.12E-02, 5.47E-02, 4.98E-02, 4.28E-02, 3.46E-02, 2.98E-02, 2.67E-02, 2.46E-02, 2.19E-02, 2.02E-02, 1.81E-02, 1.72E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] teflonMuEnOverRho = {4.80E+03, 1.66E+03, 7.57E+02, 2.40E+02, 1.04E+02, 5.33E+01, 3.08E+01, 1.28E+01, 6.41E+00, 1.80E+00, 7.22E-01, 2.02E-01, 8.67E-02, 4.94E-02, 3.47E-02, 2.52E-02, 2.34E-02, 2.42E-02, 2.58E-02, 2.76E-02, 2.84E-02, 2.85E-02, 2.84E-02, 2.77E-02, 2.68E-02, 2.56E-02, 2.45E-02, 2.26E-02, 1.98E-02, 1.81E-02, 1.69E-02, 1.61E-02, 1.50E-02, 1.43E-02, 1.35E-02, 1.31E-02};
	/**
	 * Energies [Mev] at which the coefficients were measured. 
	 */
	public static double [] CsIEnergies = {1.00000E-03, 1.03199E-03, 1.06500E-03, 1.06500E-03, 1.06854E-03, 1.07210E-03, 1.07210E-03, 1.14230E-03, 1.21710E-03, 1.21710E-03, 1.50000E-03, 2.00000E-03, 3.00000E-03, 4.00000E-03, 4.55710E-03, 4.55710E-03, 4.70229E-03, 4.85210E-03, 4.85210E-03, 5.00000E-03, 5.01190E-03, 5.01190E-03, 5.09924E-03, 5.18810E-03, 5.18810E-03, 5.27305E-03, 5.35940E-03, 5.35940E-03, 5.53401E-03, 5.71430E-03, 5.71430E-03, 6.00000E-03, 8.00000E-03, 1.00000E-02, 1.50000E-02, 2.00000E-02, 3.00000E-02, 3.31694E-02, 3.31694E-02, 3.45483E-02, 3.59846E-02, 3.59846E-02, 4.00000E-02, 5.00000E-02, 6.00000E-02, 8.00000E-02, 1.00000E-01, 1.50000E-01, 2.00000E-01, 3.00000E-01, 4.00000E-01, 5.00000E-01, 6.00000E-01, 8.00000E-01, 1.00000E+00, 1.25000E+00, 1.50000E+00, 2.00000E+00, 3.00000E+00, 4.00000E+00, 5.00000E+00, 6.00000E+00, 8.00000E+00, 1.00000E+01, 1.50000E+01, 2.00000E+01};	
	/**
	 * Photon mass attenuation coefficients (mu/rho) in [cm^2/g]
	 */
	public static double [] CsIMuOverRho = {9.234E+03, 8.653E+03, 8.098E+03, 8.339E+03, 8.281E+03, 8.224E+03, 8.387E+03, 7.344E+03, 6.413E+03, 6.569E+03, 4.132E+03, 2.114E+03, 7.880E+02, 3.836E+02, 2.752E+02, 5.174E+02, 4.851E+02, 4.510E+02, 5.637E+02, 5.296E+02, 5.268E+02, 7.511E+02, 7.193E+02, 6.881E+02, 7.453E+02, 7.196E+02, 6.875E+02, 7.923E+02, 7.323E+02, 6.761E+02, 7.268E+02, 6.448E+02, 3.071E+02, 1.711E+02, 5.815E+01, 2.686E+01, 9.045E+00, 6.923E+00, 2.122E+01, 2.687E+01, 1.719E+01, 3.027E+01, 2.297E+01, 1.287E+01, 7.921E+00, 3.677E+00, 2.035E+00, 7.290E-01, 3.805E-01, 1.818E-01, 1.237E-01, 9.809E-02, 8.373E-02, 6.769E-02, 5.848E-02, 5.110E-02, 4.644E-02, 4.123E-02, 3.721E-02, 3.616E-02, 3.622E-02, 3.673E-02, 3.838E-02, 4.030E-02, 4.492E-02, 4.867E-02};
	/**
	 * mass energy-absorption coefficient (mu_\text{en}/rho) in [cm^2/g]
	 */
	public static double [] CsIMuEnOverRho = {9.213E+03, 8.633E+03, 8.080E+03, 8.320E+03, 8.262E+03, 8.205E+03, 8.368E+03, 7.327E+03, 6.398E+03, 6.553E+03, 4.120E+03, 2.104E+03, 7.809E+02, 3.776E+02, 2.696E+02, 4.936E+02, 4.628E+02, 4.303E+02, 5.332E+02, 5.012E+02, 4.985E+02, 7.037E+02, 6.744E+02, 6.457E+02, 6.987E+02, 6.716E+02, 6.454E+02, 7.397E+02, 6.847E+02, 6.331E+02, 6.795E+02, 6.043E+02, 2.906E+02, 1.624E+02, 5.486E+01, 2.496E+01, 8.071E+00, 6.088E+00, 9.086E+00, 8.529E+00, 7.990E+00, 1.059E+01, 9.395E+00, 6.596E+00, 4.586E+00, 2.399E+00, 1.391E+00, 4.951E-01, 2.401E-01, 9.634E-02, 5.828E-02, 4.366E-02, 3.657E-02, 2.987E-02, 2.657E-02, 2.402E-02, 2.243E-02, 2.089E-02, 2.059E-02, 2.144E-02, 2.255E-02, 2.364E-02, 2.563E-02, 2.725E-02, 2.992E-02, 3.114E-02};
	/**
	 * Energies [Mev] at which the coefficients were measured. 
	 */
	public static double [] airEnergies = {0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0025, 0.003, 0.003201, 0.003202, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.662, 0.7, 0.8, 1, 1.022, 1.25, 1.5, 1.75, 2, 2.044, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100};	
	/**
	 * Air absorption coefficients in [cm^2/g]
	 */
	public static double [] airAbsoprtion = {3672, 2006, 1226, 791.6, 542.6, 281.6, 164.7, 140, 154.2, 114.8, 77.28, 54.1, 39.32, 29.35, 22.47, 17.5, 13.9, 9.168, 4.533, 2.221, 1.242, 0.7576, 0.4942, 0.2438, 0.1395, 0.08919, 0.06253, 0.0473, 0.0382, 0.03254, 0.02894, 0.0266, 0.02509, 0.02356, 0.02305, 0.02382, 0.02494, 0.02588, 0.02674, 0.02787, 0.02874, 0.02918, 0.0295, 0.02962, 0.02967, 0.02962, 0.02954, 0.02937, 0.02933, 0.0292, 0.02883, 0.02789, 0.02778, 0.02665, 0.02548, 0.02442, 0.02347, 0.02331, 0.02185, 0.02058, 0.01957, 0.01872, 0.01803, 0.01743, 0.01688, 0.01641, 0.01603, 0.01568, 0.01516, 0.01454, 0.01385, 0.01347, 0.01329, 0.01319, 0.01299, 0.01287, 0.01277, 0.01271, 0.01275, 0.01278, 0.01277, 0.0128, 0.01281, 0.01281, 0.01279, 0.0127};
	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
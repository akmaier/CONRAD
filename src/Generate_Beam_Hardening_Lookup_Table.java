import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients;
import edu.stanford.rsl.conrad.physics.XRaySpectrum;
import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients.Material;
import edu.stanford.rsl.conrad.utils.BilinearInterpolatingDoubleArray;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.LinearInterpolatingDoubleArray;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.plugin.PlugIn;


public class Generate_Beam_Hardening_Lookup_Table implements PlugIn{

	public Generate_Beam_Hardening_Lookup_Table (){
		
	}
	
	@Override
	public void run(String arg) {
		try {
			// get Stats for the spectrum
			double min = UserUtil.queryDouble("Minimum [keV]", 10);
			double max = UserUtil.queryDouble("Maximum [keV]", 150);
			double delta = UserUtil.queryDouble("Delta [keV]", 0.5);
			double maxWater = UserUtil.queryDouble("Maximal observed accumulated attenuation in the water corrected image: ", 15);
			double maxPassedMaterial = UserUtil.queryDouble("Maximal observed accumulated attenuation in the hard material: ", 3);
			double stepSize = UserUtil.queryDouble("Step size for the sampling of the lookup table: ", 0.01);
			// Query the materials
			Material material = UserUtil.queryMaterial("Please Select the soft Material:", "Material Selection");
			LinearInterpolatingDoubleArray lutSoft = EnergyDependentCoefficients.getPhotonMassAttenuationLUT(material);
			material = UserUtil.queryMaterial("Please Select the hard Material:", "Material Selection");
			LinearInterpolatingDoubleArray lutHard = EnergyDependentCoefficients.getPhotonMassAttenuationLUT(material);
			// sample the x-ray spectrum 
			int steps = (int) ((max - min) / delta);
			double [] energies = new double [steps];
			for (int i = 0; i < steps; i ++){
				energies[i] = min + (i*delta);
			}
			double [] spectrum = XRaySpectrum.generateXRaySpectrum(energies, 120, "W", 1, 1, 12, 2.38, 3.06, 2.66, 10.5);
			BilinearInterpolatingDoubleArray beamHardeningLut = EnergyDependentCoefficients.getBeamHardeningLookupTable(maxWater, maxPassedMaterial, stepSize, energies, spectrum, lutSoft, lutHard);
			Configuration.getGlobalConfiguration().setBeamHardeningLookupTable(beamHardeningLut);
			VisualizationUtil.showImageProcessor(beamHardeningLut.toFloatProcessor(2048, 512), "Generated Beam Hardening Lookup Table");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

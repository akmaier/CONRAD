import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients;
import edu.stanford.rsl.conrad.physics.EnergyDependentCoefficients.Material;
import edu.stanford.rsl.conrad.utils.LinearInterpolatingDoubleArray;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.gui.Plot;
import ij.plugin.PlugIn;


public class Generate_Photon_Mass_Attenuation_Plot implements PlugIn{

	public Generate_Photon_Mass_Attenuation_Plot (){
		
	}
	
	@Override
	public void run(String arg) {
		try {
			double min = UserUtil.queryDouble("Minimum [keV]", 10);
			double max = UserUtil.queryDouble("Maximum [keV]", 150);
			double delta = UserUtil.queryDouble("Delta [keV]", 0.5);
			Material material = UserUtil.queryMaterial("Please Select the Material:", "Material Selection");
			LinearInterpolatingDoubleArray lut = EnergyDependentCoefficients.getPhotonMassAttenuationLUT(material);
			int steps = (int) ((max - min) / delta);
			double [] energies = new double [steps];
			double [] coefficients = new double[energies.length];
			for (int i = 0; i < steps; i ++){
				energies[i] = min + (i*delta);
				coefficients[i] = (lut.getValue(energies[i] / 1000));
			}
			//DoubleArrayUtil.log(spectrum);
			Plot plot = new Plot(material + " photon mass attenuation coefficient", "Energy", "ln (coefficient)", energies, coefficients);
			plot.show();
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
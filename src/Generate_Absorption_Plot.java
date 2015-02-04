/*
 * Copyright (C) 2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.gui.Plot;
import ij.plugin.PlugIn;


public class Generate_Absorption_Plot implements PlugIn{

	public Generate_Absorption_Plot (){
		
	}
	
	@Override
	public void run(String arg) {
		try {
			double min = UserUtil.queryDouble("Minimum [keV]", 10);
			double max = UserUtil.queryDouble("Maximum [keV]", 150);
			double delta = UserUtil.queryDouble("Delta [keV]", 0.5);
			Object materialString = UserUtil.chooseObject("Please select a material: ", "Material Selection", MaterialsDB.getMaterials(), "water");
			Material material = MaterialsDB.getMaterial(materialString.toString());
			
			int steps = (int) ((max - min) / delta);
			double [] energies = new double [steps];
			double [] absorption =  new double [steps];
			
			for (int i = 0; i < steps; i ++){
				energies[i] = min + (i*delta);
				absorption[i] = material.getAttenuation(energies[i], AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION) / material.getDensity();
			}
			
			Plot plot = new Plot("Absorption Spectrum of " + material, "Energy [keV]", "mu / rho [cm^2/g]", energies, absorption);
			plot.show();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}


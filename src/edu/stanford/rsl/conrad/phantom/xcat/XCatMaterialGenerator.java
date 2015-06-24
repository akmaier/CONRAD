/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.xcat;

import java.util.HashMap;

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

public abstract class XCatMaterialGenerator {



	/**
	 * Get the material from its name.
	 * @param name the name
	 * @return the material
	 */
	public static Material generateFromMaterialName(String name){
		Double value = getMaterialDensityLUT().get(name);
		if (value == null) System.out.println("Error value was null: " + name);
		Material material = MaterialsDB.getMaterialWithName("Water");
		material.setName(name + " (water w rho=" + value + " [g/cm^3])");
		if (name.contains("Bone")) material = MaterialsDB.getMaterialWithName("Bone");
		if (name.contains("Marrow")) material = MaterialsDB.getMaterialWithName("Bonemarrow");
		if (name.contains("Catheter")) material = MaterialsDB.getMaterialWithName("copper");
		material.setDensity(value);
		// using custom contrast agent from tutorial code.
		// in this material the density is already set correctly.
		if (name.contains("Coronary Artery")) {
			if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ONLY_LEFT_ARTERY_TREE_CONTRASTED)!= null) {
				if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ONLY_LEFT_ARTERY_TREE_CONTRASTED).equals("true")) {
				}
			}else{
				material = MaterialsDB.getMaterialWithName("Ultravist370");
			}
		}
		
		return material;
	}

	/**
	 * The material-name lookup table. 
	 * @return the material-name lut
	 */
	public static HashMap<String, Double> getMaterialDensityLUT(){
		HashMap<String, Double> map = new HashMap<String, Double>();
		map.put("Body (water)", 1.0);
		map.put("Adipose (fat)", 0.920000);
		map.put("Lung", 0.300000);
		map.put("Air", 0.0);
		map.put("Spine Bone", 1.420000);
		map.put("Rib Bone", 1.620000);
		map.put("Bone", 1.620000);
		map.put("Skull Bone", 1.610000);
		map.put("Blood", 1.060000);
		map.put("Blood Kidney Vein", 1.060000);
		map.put("Blood Liver Vein", 1.060000);
		map.put("Blood Vein", 1.060000);
		map.put("Blood Pul Vein", 1.060000);
		map.put("Blood Vena Cava", 1.060000);
		map.put("Blood Artery", 1.060000);
		map.put("Blood Pul Artery", 1.060000);
		map.put("Blood Kid Artery", 1.060000);
		map.put("Blood (Aorta)", 1.060000);
		map.put("Blood (LV)", 1.060000);
		map.put("Coronary Artery", 4.000000);
		map.put("Heart", 1.050000);
		if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_CONTRAST_LEFT_VENTRICLE) != null) {
			if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_CONTRAST_LEFT_VENTRICLE).equals("true")){
				map.put("Blood (Aorta)", 2.00000);
				map.put("Blood (LV)", 2.50000);
				map.put("Heart", 1.50000);
			}
		}
		map.put("Muscle", 1.16);
		map.put("Kidney", 1.050000);
		map.put("Liver", 1.060000);
		map.put("Lymph", 1.030000);
		// Changed from
		// map.put("Bone Marrow", 1.120000);
		// to
		map.put("Bone Marrow", 0.98);
		// densities for red marrow and yellow marrow are 1.03 and 0.98. We chose red marrow here 
		// based on Gary Beaupre's suggestion:
		// H. Q. Woodard, and D. R. White. The composition of body tissues
		// 1986, The British Journal of Radiology, 59, 1209-1219 DECEMBER 1986
		// This is consistent with
		// ICRU Report 46. Photon, Electron, Proton and Neutron Interaction Data for Body Tissues. Bethesda, MD: International Commission on Radiation Units and Measurements; 1992.
		map.put("Pancreas", 1.040000);
		map.put("Spleen", 1.060000);
		//map.put("Spleen", 1.360000);
		map.put("Intestine", 1.030000);
		map.put("Cartilage", 1.100000);
		map.put("Brain", 1.040000);
		map.put("Heart Lesion", 2.0000);
		map.put("Catheter Material", 5.0);
		
		String value = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_USE_MRI_CONTRAST_SETTINGS);
		if (value != null){
			if (Boolean.parseBoolean(value)){
				map.put("Body (water)",         0.16); /// ?
				map.put("Adipose (fat)",        0.10); ///
				map.put("Lung",                 0.04); ///
				map.put("Air",                  0.00); ///
				map.put("Spine Bone",           0.00); ///
				map.put("Rib Bone",             0.00); ///
				map.put("Bone",                 0.00); ///
				map.put("Skull Bone",           0.00); ///
				map.put("Blood",                0.25); /// ?
				map.put("Blood Kidney Vein",    0.27); /// ?
				map.put("Blood Liver Vein",     0.35); /// ?
				map.put("Blood Vein",           0.25); /// ?
				map.put("Blood Pul Vein",       0.24); ///
				map.put("Blood Vena Cava",      0.29); ///
				map.put("Blood Artery",         0.32); /// ?
				map.put("Blood Pul Artery",     0.31); ///
				map.put("Blood Kid Artery",     0.29); /// ?
				map.put("Blood (Aorta)",        0.32); ///
				map.put("Blood (LV)",           0.42); ///
				map.put("Coronary Artery",      0.35); ///
				map.put("Heart",                0.12); ///
				if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_CONTRAST_LEFT_VENTRICLE) != null) {
				   if (Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_CONTRAST_LEFT_VENTRICLE).equals("true")){
				       map.put("Blood (Aorta)", 0.42); ///
				       map.put("Blood (LV)",    0.57); ///
				       map.put("Heart",         0.18); ///
				   }
				}
				map.put("Muscle",               0.22); /// ?
				map.put("Kidney",               0.37); ///
				map.put("Liver",                0.26); ///
				map.put("Lymph",                0.50); /// ?
				map.put("Bone Marrow",          0.31); ///
				map.put("Pancreas",             0.32); /// ?
				map.put("Spleen",               0.34); /// ?
				map.put("Intestine",            0.00); /// not in ROI
				map.put("Cartilage",            0.22); /// ?
				map.put("Brain",                0.00); /// not in ROI
				map.put("Heart Lesion",         0.00); /// not needed
				map.put("Catheter Material",    0.00); /// not needed
			}
		}
		
		return map;
	}
}

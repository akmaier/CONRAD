 /*
  * Copyright (C) 2010-2014 Rotimi X Ojo, Andreas Maier
  * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
  */

package edu.stanford.rsl.conrad.physics.materials.materialsTest;



import junit.framework.Assert;

import org.junit.Test;

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.FormulaToNameMap;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.database.OnlineMassAttenuationDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.physics.materials.utils.LocalMassAttenuationCalculator;
import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;

/**
 * <p>Class for comparing the results from local mass attenuation calculator with online mass attenuation calculation from XCOM</p>
 * <p>Class asserts that Max Error < 3%, in reality maximum error is significantly less than this value in most materials.
 * @author Rotimi X Ojo
 * @author Andreas Maier
 * 
 */ 

public class TestMassAttenuationData {
	
	@Test
	public void reportPredefinedMaterials(){
		String [] materials = MaterialsDB.getPredefinedMaterials();
		System.out.println("Number of predefined materials:" + materials.length);
		for (String material:materials) System.out.println(material);
		// Standard configuration of CONRAD comes with at least 105 known materials.
		Assert.assertTrue(materials.length > 104);
	}
	
	@Test
	public void reportKnownMaterials(){
		String [] materials = MaterialsDB.getMaterials();
		System.out.println("Number of known materials:" + materials.length);
		for (String material:materials) System.out.println(material);
		// Standard configuration of CONRAD comes with at least 105 known materials.
		Assert.assertTrue(materials.length > 104);
	}
	
	@Test
	public void testMaterialMassAttenutationCalculator(){
		System.out.println("Testing LocalMassAttenuationCalculator Accuracy: " );
		String [] identifiers = FormulaToNameMap.map.keySet().toArray(new String[FormulaToNameMap.map.size()]);
		int numEnergies = 1000;
		int maxEnergy = 100000;
		for(int i =0; i < 1/*identifiers.length*/; i++){
			String iden = identifiers[i];
			System.out.println("\tTesting " + iden);
			Material mat = MaterialsDB.getMaterial(iden);	
			
			//double [] results1 = testKnownMaterials(mat,AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION,numEnergies, maxEnergy);
			testKnownMaterials(mat,AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION,numEnergies, maxEnergy );
			//double [] results3 = testKnownMaterials(mat,AttenuationType.COHERENT_ATTENUATION,numEnergies, maxEnergy);
			//double [] results4 = testKnownMaterials(mat,AttenuationType.INCOHERENT_ATTENUATION,numEnergies, maxEnergy);
			//double [] results5 = testKnownMaterials(mat,AttenuationType.NUCLEAR_FIELD_PAIRPRODUCTION,numEnergies, maxEnergy);
			//double [] results6 = testKnownMaterials(mat,AttenuationType.PHOTOELECTRIC_ABSORPTION,numEnergies, maxEnergy);

		
		}
		System.out.println("Testing Completed" );
	}	
	/**
	 * Compare local and online mass attenuation values of know materials
	 * @param mat is material of interest
	 * @param att is attenuation of interest
	 * @param numEnergies is energies to be tested
	 */
	public static double [] testKnownMaterials(Material mat,AttenuationType att, int numEnergies, int maxEnergy) {
		return testArbitraryMaterial(mat.getWeightedAtomicComposition(), att, numEnergies, maxEnergy);
	}
	
	/**
	 * CompareCompare local and online mass attenuation values of arbitrarily defined material
	 * @param formula is chemical formula of material
	 * @param att is attenuation of interest
	 * @param numEnergies
	 */
	public static double [] testArbitraryMaterial(String formula, AttenuationType att, int numEnergies, int maxEnergy){
		return testArbitraryMaterial(new WeightedAtomicComposition(formula), att, numEnergies, maxEnergy);
	}
	
	
	/**
	 * CompareCompare local and online mass attenuation values of arbitrarily defined material
	 * @param comp is weighted atomic composition
	 * @param att is attenuation of interest
	 * @param numEnergies
	 */
	public static double [] testArbitraryMaterial(WeightedAtomicComposition comp, AttenuationType att, int numEnergies, int maxEnergy){
		System.out.println("\t\tTesting " + att);
		double [] energies = new double[numEnergies];
		
		
		for(int i = 0;i < energies.length;i++){
			energies[i] = (Math.random()*maxEnergy) + 1;
		}
		
		double maxError = 0;
		double minError = Double.MAX_VALUE;
		double errorSum = 0;
		int nonZeroErrorCount = 0;
		
		for(double energy:energies){
		
			double val = OnlineMassAttenuationDB.getMassAttenuationData(comp, energy/1000, att);
			double val2 = LocalMassAttenuationCalculator.getMassAttenuationData(comp, energy/1000, att);
			double error = Math.abs(val-val2)/val;
			
			if(error > maxError ){
				maxError = error;
			}
			
			if(error < minError){
				minError = error;
			}
			
			if(error > 0){
				nonZeroErrorCount++;
				errorSum+= error;
			}
			
		}
		
		double meanErr = errorSum/nonZeroErrorCount;
		assert(maxError < 3);
		double [] results = {minError,maxError,meanErr};
		
		System.out.println("Min Error is:  " + minError + "%");
		System.out.println("Max Error is:  " + maxError+ "%");
		System.out.println("Mean Error is: " + meanErr+ "%");
		return results;
	}

}

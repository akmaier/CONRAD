/*
 * Copyright (C) 2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.physics.materials.Mixture;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.MaterialUtils;
import edu.stanford.rsl.conrad.physics.materials.utils.WeightedAtomicComposition;


/**
 * Class to create a custom material. Note that CONRAD is able to model also very advanced materials in very high detail.
 *  
 * In this example, we create four new materials in the material database.
 * We create <a href="http://bayerimaging.com/products/ultravist/">Ultravist(C)</a> in concentration 150, 240, 300, and 370.
 * The physical properties of these mixtures are taken from <a href="http://www.rxlist.com/ultravist-drug.htm">www.rxlist.com</a>.
 * The chemical formula for Iopromide is taken from <a href="http://en.wikipedia.org/wiki/Iopromide">Wikipedia</a>.
 * 
 * @author Andreas Maier
 *
 */
public class CreateCustomMaterial {

	
	/**
	 * Here we create ULTRAVIST 150, 240, 300, 370.
	 * Chemical formula, density etc. are taken from<br>
	 * http://www.rxlist.com/ultravist-drug.htm<br>
	 * and<BR>
	 * http://en.wikipedia.org/wiki/Iopromide
	 *
	 * @param args no arguments here.
	 */
	public static void main(String[] args) {
		// from http://en.wikipedia.org/wiki/Iopromide
		String formularIopromideString = "C18H24I3N3O8";
		// @37 deg celsius
		double additionalDensityInWater = 0.529816;
		// from http://www.rxlist.com/ultravist-drug.htm
		double [] additionalGramsOfIopromide = {.3117,	.49872,	.6234,	.76886};
		double [] densityAt37DegCelsius = {1.157,	1.255,	1.322,	1.399};
		String suffix[] = {"150", "240", "300", "370"};
		
		// This would be plain iopromide:
		WeightedAtomicComposition wacIopromide = new WeightedAtomicComposition(formularIopromideString);
		double molarMassIopromide = MaterialUtils.computeMolarMass(wacIopromide);
		MaterialUtils.newMaterial("Iopromide",additionalDensityInWater,wacIopromide);
		
		WeightedAtomicComposition water = new WeightedAtomicComposition("H2O");
		double molarMassWater = MaterialUtils.computeMolarMass(water);
		double waterParticlesIn1Gram = 1 / molarMassWater;
		
		// Create compositions as used in clinical practice:
		for(int i = 0; i < additionalGramsOfIopromide.length; i++) {
			WeightedAtomicComposition wacUltravist = new WeightedAtomicComposition("H2O", waterParticlesIn1Gram);
			double iopromideParticles = additionalGramsOfIopromide[i] / molarMassIopromide;
			wacUltravist.add(formularIopromideString, iopromideParticles);
			//Material ultravist = MaterialUtils.newMaterial(,densityAt37DegCelsius[i],wacUltravist);
			Mixture iopromideSolution = new Mixture();
			iopromideSolution.setDensity(densityAt37DegCelsius[i]);
			iopromideSolution.setName("Ultravist"+suffix[i]);
			iopromideSolution.setWeightedAtomicComposition(wacUltravist);
			MaterialsDB.put(iopromideSolution);
		}
	}

}

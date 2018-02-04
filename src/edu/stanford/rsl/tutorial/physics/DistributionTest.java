/*
 * Copyright (C) 2014 - Andreas Maier, Oliver Taubmann, Fabian Rückert 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;

/**
 * Simple example to show how Monte Carlo simulation works. Here, we simulate hitting a virtual boundary between two times the same material.
 * We use the famous formula, that is the inversion of the monochromatic absorption law to predict the path length of the photon until interaction.
 * (Here we simply assume that the interaction will be a total absorption.)<BR>
 * When we hit the boundary now, we need to draw again. This is done by doing another random experiment.<br>
 * In the end, we compare mean values and standard deviations in a single material process and a process with boundary and show numerically that the distribution does not change.
 * 
 * @author akmaier
 *
 */
public class DistributionTest {

	/**
	 * The inversion of the monochromatic absorption law
	 * @param r the random number between 0 and 1
	 * @param mu the absorption coefficient
	 * @return the path length
	 */
	public static double famousFormula(double r, double mu) {
		return -Math.log(r) / mu;
	}

	public static void main(String[] args) {
		// get water properties
		Material water = MaterialsDB.getMaterial("water");

		// number of trials for comparison
		int N = 100000000;
		// result buffers
		double[] result = new double[N];
		double[] result2 = new double[N];
		// absorption coefficient
		double mu = water.getAttenuation(55, AttenuationType.TOTAL_WITH_COHERENT_ATTENUATION);
		// process without intersection
		for (int i = 0; i < N; i++) {
			result[i] = famousFormula(Math.random(), mu);
		}
		double mean = DoubleArrayUtil.computeMean(result);
		double stabw = DoubleArrayUtil.computeStddev(result, mean);
		System.out.println("Mean " + mean + "\nStabw " + stabw);

		// process with intersection at len:
		double len = 2;
		for (int i = 0; i < N; i++) {
			// probability for interaction (p(l))
			double probl = Math.random();
			// resulting path l
			result2[i] = famousFormula(probl, mu);
			// boundary exceeded
			if (result2[i] > len) {
				// new trial 
				result2[i] = len + famousFormula(Math.random(), mu);
			}
		}

		double mean2 = DoubleArrayUtil.computeMean(result2);
		double stabw2 = DoubleArrayUtil.computeStddev(result2, mean2);
		System.out.println("Mean " + mean2 + "\nStabw " + stabw2);
	}

}

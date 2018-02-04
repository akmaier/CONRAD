package edu.stanford.rsl.tutorial.physics;

import java.util.Random;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;

//********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************

/**
 * Does the Monte Carlo sampling using a given Random object (for example ThreadLocalRandom)
 * @author Fabian Rückert
 */
public class XRayTracerSampling {

	public static final double eV = 1.60217656535e-19;
	private static final double electron_mass_c2 = 0.51099906 * eV * 1000000;
	public static final double electronMass = 9.10938291e-31; // kg
	public static final double c = 299792458; // m/s
	public static final double electronRadius = 2.8179E-15; // m
	private Random myRandom;

	public XRayTracerSampling(Random random) {
		this.myRandom = random;
	}

	public double random() {
		return myRandom.nextDouble();
	}

	/**
	 * Sample the scattering energy/angle by using the method from Geant4.
	 * @see http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/PhysicsReferenceManual/fo/PhysicsReferenceManual.pdf
	 * @see http://geant4www.triumf.ca/lxr/source//processes/electromagnetic/standard/src/G4KleinNishinaCompton.cc
	 * @param energy The energy of the incident photon in eV
	 * @param dir The incident direction, outputs the resulting direction
	 * @return The resulting energy
	 */
	public double sampleComptonScatteringGeant4(double energy, SimpleVector dir) {

		//convert to joule
		double gamEnergy0 = energy * XRayTracerSampling.eV;

		//Geant4 Code ---------------------
		double E0_m = gamEnergy0 / electron_mass_c2;

		double epsilon, epsilonsq, onecost, sint2, greject;

		double epsilon0 = 1. / (1. + 2. * E0_m);
		double epsilon0sq = epsilon0 * epsilon0;
		double alpha1 = -Math.log(epsilon0);
		double alpha2 = 0.5 * (1. - epsilon0sq);

		do {
			if (alpha1 / (alpha1 + alpha2) > random()) {
				epsilon = Math.exp(-alpha1 * random()); // epsilon0**r
				epsilonsq = epsilon * epsilon;

			} else {
				epsilonsq = epsilon0sq + (1. - epsilon0sq) * random();
				epsilon = Math.sqrt(epsilonsq);
			}

			onecost = (1. - epsilon) / (epsilon * E0_m);
			sint2 = onecost * (2. - onecost);
			greject = 1. - epsilon * sint2 / (1. + epsilonsq);

		} while (greject < random());

		double cosTeta = 1. - onecost;
		double sinTeta = Math.sqrt(sint2);
		double Phi = 2 * Math.PI * random();
		double dirx = sinTeta * Math.cos(Phi), diry = sinTeta * Math.sin(Phi), dirz = cosTeta;

		SimpleVector gamDirection1 = new SimpleVector(dirx, diry, dirz);

		double gamEnergy1 = epsilon * gamEnergy0;
		//Geant4 Code ---------------------

		// transform scattered photon direction vector to world coordinate system		
		SimpleVector resultdir = transformToWorldCoordinateSystem(gamDirection1, dir);
		dir.setSubVecValue(0, resultdir);

		//convert back to eV
		return gamEnergy1 / XRayTracerSampling.eV;
	}

	/**
	 * Sample compton scattering, calculate the resulting energy and direction.
	 * @param energyEV The incident photon energy in eV
	 * @param dir The direction of the photon, the direction will be changed to the new direction
	 * @return The resulting energy of the photon
	 */
	public double sampleComptonScattering(double energyEV, SimpleVector dir) {
		double theta = getComptonAngleTheta(energyEV * eV);

		double phi = 2 * Math.PI * random();
		double dirx = Math.sin(theta) * Math.cos(phi);
		double diry = Math.sin(theta) * Math.sin(phi);
		double dirz = Math.cos(theta);

		SimpleVector gamDirection1 = new SimpleVector(dirx, diry, dirz);
		gamDirection1.normalizeL2();

		// transform scattered photon direction vector to world coordinate system
		SimpleVector ret = transformToWorldCoordinateSystem(gamDirection1, dir);
		dir.setSubVecValue(0, ret);

		return getScatteredPhotonEnergy(energyEV, theta);
	}

	/**
	 * Calculates the resulting energy for the comtpon effect.
	 * @param energyEV The incident photon energy in eV
	 * @param theta The scattering angle
	 * @return	The energy of the photon after scattering
	 */
	public static double getScatteredPhotonEnergy(double energyEV, double theta) {
		double energyJoule = energyEV * eV;

		return (energyJoule / (1 + (energyJoule / (electronMass * c * c)) * (1 - Math.cos(theta)))) / eV;
	}

	/**
	 * The Klein-Nishina differential cross section
	 * @param energyJoule The incident photon energy in joule
	 * @param theta	The scattering angle
	 * @return	The Klein-Nishina differential cross section
	 */
	private static double kleinNishinaDifferentialCrossSection(double energyJoule, double theta) {
		double PE0 = 1 / (1 + (energyJoule / (electronMass * c * c)) * (1 - Math.cos(theta)));

		return ((electronRadius * electronRadius) / 2) * PE0
				* (1 - PE0 * Math.sin(theta) * Math.sin(theta) + PE0 * PE0);
	}

	/**
	 * The angular distribution of scattered photons
	 * @param energyJoule The incident photon energy in joule
	 * @param theta The scattering angle
	 * @return The relative probability for scattering with this angle
	 */
	public static double comptonAngleProbability(double energyJoule, double theta) {
		return kleinNishinaDifferentialCrossSection(energyJoule, theta) * 2 * Math.PI * Math.sin(theta);
	}

	/**
	 * Sample the scattering angle for compton effect using simple rejection sampling
	 * @param energyJoule The incident photon energy in joule
	 * @return	A random scattering angle (radians) following the klein-nishina distribution
	 */
	public double getComptonAngleTheta(double energyJoule) {
		// find approximate maximum
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < 180; ++i) {
			double value = comptonAngleProbability(energyJoule, Math.toRadians(i));

			if (value < min) {
				min = value;
			}
			if (value > max) {
				max = value;
			}
		}

		//rejection sampling
		while (true) {
			double r1 = random() * Math.PI;
			double r2 = random() * max;

			if (r2 < comptonAngleProbability(energyJoule, r1)) {
				return r1;
			}
		}

	}

	/**
	 * Rotate the reference frame such that the original Z-axis will lie in the direction of ref
	 * @see http://proj-clhep.web.cern.ch/proj-clhep/manual/UserGuide/VectorDefs/node49.html
	 * @param v
	 * @param ref
	 * @return The rotated vector
	 */
	public static SimpleVector transformToWorldCoordinateSystem(SimpleVector v, SimpleVector ref) {

		double ux = ref.getElement(0);
		double uy = ref.getElement(1);
		double uz = ref.getElement(2);

		double uPar = ux * ux + uy * uy;
		if (uPar > 0) {
			uPar = Math.sqrt(uPar);
			SimpleMatrix mat = new SimpleMatrix(3, 3);
			mat.setColValue(0, new SimpleVector(ux * uz / uPar, uy * uz / uPar, -uPar));
			mat.setColValue(1, new SimpleVector(-uy / uPar, ux / uPar, 0));
			mat.setColValue(2, ref);
			SimpleVector a = SimpleOperators.multiply(mat, v);
			return a;
		} else if (ref.getElement(2) < 0) {
			//rotate by 180 degrees about the y axis
			return new SimpleVector(-v.getElement(0), v.getElement(1), -v.getElement(2));
		}
		return v;
	}

	public double getDistanceUntilNextInteractionCm(Material material, double energyEV) {
		double photo = material.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
		double compton = material.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);

		//use the sum of compton effect and photoelectric absorption instead of AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION
		//because only these effects are simulated
		return getDistanceUntilNextInteractionCm(photo, compton);
	}

	/**
	 * @param photo The attenuation for the photoelectric effect
	 * @param compton The attenuation for the compton effect
	 * @return The distance the photon travels
	 */
	public double getDistanceUntilNextInteractionCm(double photo, double compton) {
		return -Math.log(random()) / (photo + compton);
	}
}

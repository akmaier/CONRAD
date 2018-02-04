package edu.stanford.rsl.tutorial.physics;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Random;

import ij.process.FloatProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.physics.materials.utils.AttenuationType;

/**
 * Simple single-threaded Monte Carlo Simulation of photons passing through one material.
 * 
 * @author Fabian Rückert
 * 
 */
public class SingleMaterialMonteCarlo {

	// settings
	public static final int numRays = 10000;
	public static final double scale = 1;// the zoomlevel

	// number of samples for the sampling test
	public static final int numSamples = 10000;

	// constants
	public static final double electronRadius = 2.8179E-15; // m
	public static final double electronMass = 9.10938291e-31; // kg
	public static final double c = 299792458; // m/s
	public static final Material material = MaterialsDB.getMaterial("water");
	public static final double eV = 1.60217656535e-19;

	// variables
	private int numDraws = 0;
	private int sumMisses = 0;

	ArrayList<Double> pathlengths = new ArrayList<Double>();
	ArrayList<Double> xs = new ArrayList<Double>();
	ArrayList<Double> ys = new ArrayList<Double>();
	ArrayList<Double> zs = new ArrayList<Double>();

	XRayTracerSampling sampler = new XRayTracerSampling(new Random());

	public SingleMaterialMonteCarlo() {
		double energyEV = 1000000;

		System.out.println("energy " + energyEV / 1000000 + " MeV");

		double photo = material.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
		double compton = material.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);
		System.out.println("cross section photo: " + photo + "cm" + " compton " + compton + "cm");
		System.out.println("mean free path compton " + 1 / compton + "cm");
		System.out.println("mean free path photo " + 1 / photo + "cm");

		double totalCrossSection = material.getAttenuation(energyEV / 1000,
				AttenuationType.TOTAL_WITHOUT_COHERENT_ATTENUATION);
		System.out.println("total attenuation " + 1 / totalCrossSection + "cm");
		System.out.println("total attenuation sum " + 1 / (photo + compton) + "cm");

		raytrace(energyEV, numRays);

		double mean = XRayTracer.computeMean(pathlengths);
		double stddev = XRayTracer.computeStddev(pathlengths, mean);
		System.out.println("pathlength = " + mean + " +- " + stddev);

		mean = XRayTracer.computeMean(xs);
		stddev = XRayTracer.computeStddev(xs, mean);
		System.out.println("x = " + mean + " +- " + stddev);

		mean = XRayTracer.computeMean(ys);
		stddev = XRayTracer.computeStddev(ys, mean);
		System.out.println("y = " + mean + " +- " + stddev);

		mean = XRayTracer.computeMean(zs);
		stddev = XRayTracer.computeStddev(zs, mean);
		System.out.println("z = " + mean + " +- " + stddev);

		// rayTrace(800000, numRays);

		// visualize the klein-nishina angle distribution and the rejection
		// sampling for different energies
		//		 visualizeKleinNishina(10000000*eV);
		visualizeKleinNishina(140000 * eV);

		// visualizeKleinNishina(2.75*eV);
		//
		// visualizeKleinNishina(55000*eV);

	}

	private void raytrace(double energyEV, int numRays) {
		Grid2D grid = new Grid2D(600, 500);
		FloatProcessor imp;
		imp = new FloatProcessor(grid.getWidth(), grid.getHeight());
		imp.setPixels(grid.getBuffer());

		//		 SimpleVector startPosition = new SimpleVector(40,grid.getHeight()/2/scale, 0);
		SimpleVector startPosition = new SimpleVector(0, 0, 0);

		for (int i = 0; i < numRays; ++i) {
			followRay(startPosition.clone(), new SimpleVector(1, 0, 0), energyEV, imp, 0, 0);
		}
		imp.drawString(grid.getWidth() / scale + "cm", grid.getWidth() / 2 - 20, grid.getHeight() - 10, Color.WHITE);

		grid.show("Energy: " + energyEV + "eV, Material: " + material.getName());

		System.out.println("Rejection sampling: average misses per draw: " + (float) sumMisses / (float) numDraws);

		numDraws = 0;
		sumMisses = 0;
	}

	private void followRay(SimpleVector pos, SimpleVector dir, double energyEV, FloatProcessor imp, int scatterCount,
			double totalDistance) {
		if (energyEV <= 1 || scatterCount > 20000) {
			System.out.println("energy low, times scattered: " + scatterCount);
			return;
		}
		// follow ray until next interaction point
		SimpleVector oldPos = pos.clone();
		double dist = sampler.getDistanceUntilNextInteractionCm(material, energyEV);
		pos.add(dir.multipliedBy(dist));
		pathlengths.add(dist);
		// draw the entire path
		// imp.drawLine((int)(scale*oldPos.getElement(0)), (int)(scale*oldPos.getElement(1)), (int)(scale*pos.getElement(0)), (int)(scale*pos.getElement(1)));
		// draw interaction points only
		imp.drawDot((int) (scale * pos.getElement(0)), (int) (scale * pos.getElement(1)));

		// choose compton or photoelectric effect
		double photo = material.getAttenuation(energyEV / 1000, AttenuationType.PHOTOELECTRIC_ABSORPTION);
		double compton = material.getAttenuation(energyEV / 1000, AttenuationType.INCOHERENT_ATTENUATION);

		if (sampler.random() * (photo + compton) <= photo) {
			// photoelectric absorption
			energyEV = 0;
			// System.out.println("absorbed after " + scatterCount + " collisions");
			xs.add(pos.getElement(0));
			ys.add(pos.getElement(1));
			zs.add(pos.getElement(2));

			return;
		} else {
			// compton scattering

			energyEV = sampler.sampleComptonScattering(energyEV, dir);

			// send new ray
			followRay(pos, dir, energyEV, imp, scatterCount + 1, totalDistance + dist);
		}

	}

	private void visualizeKleinNishina(double energyJoule) {
		int gridWidth = 600;
		int gridHeight = 500;
		double maxAngle = 360;

		Grid2D grid = new Grid2D(gridWidth, gridHeight);

		FloatProcessor fp = new FloatProcessor(grid.getWidth(), grid.getHeight());
		fp.setPixels(grid.getBuffer());

		// find max value
		double max = 0;
		for (int i = 0; i < maxAngle; ++i) {
			double value = XRayTracerSampling.comptonAngleProbability(energyJoule, Math.toRadians(i));

			if (value > max) {
				max = value;
			}

		}

		fp.drawLine((int) 0, grid.getHeight() / 2, grid.getWidth(), grid.getHeight() / 2);
		fp.drawLine(grid.getWidth() / 2, 0, grid.getWidth() / 2, grid.getHeight());
		fp.drawOval((grid.getWidth() - grid.getHeight()) / 2, 0, grid.getHeight(), grid.getHeight());
		double lastX = 0;
		double lastY = 0;

		// draw angle distribution
		for (int i = 0; i < maxAngle; ++i) {
			double value = XRayTracerSampling.comptonAngleProbability(energyJoule, Math.toRadians(i));

			SimpleMatrix m = Rotations.createBasicZRotationMatrix(Math.toRadians(i));
			SimpleVector v = new SimpleVector(((value / max) * ((double) grid.getHeight() / 2.d)), 0, 0);

			SimpleVector r = SimpleOperators.multiply(m, v);

			double x = grid.getWidth() / 2 + r.getElement(0);
			double y = grid.getHeight() / 2 + r.getElement(1);
			if (i > 0)
				fp.drawLine((int) lastX, (int) lastY, (int) x, (int) y);
			lastX = x;
			lastY = y;
		}

		grid.show("Normalized Klein-Nishina cross-section as a function of scatter angle (" + energyJoule / eV + "eV)");

		grid = new Grid2D(gridWidth, gridHeight);

		fp = new FloatProcessor(grid.getWidth(), grid.getHeight());
		fp.setPixels(grid.getBuffer());

		// draw histogram with rejection sampling of the klein-nishina
		// distribution
		int[] angles = new int[grid.getWidth()];
		for (int i = 0; i < numSamples; ++i) {
			double angle = Math.toDegrees(sampler.getComptonAngleTheta(energyJoule));
			int pos = (int) (angle * grid.getWidth() / 360.d);
			angles[pos] += 1;
		}
		double max2 = 0;
		for (int i = 0; i < angles.length; ++i) {
			if (angles[i] > max2) {
				max2 = angles[i];
			}
		}
		for (int i = 0; i < angles.length; ++i) {
			double x = i;
			double y = ((angles[i]) * (grid.getHeight() / (max2)));
			fp.drawLine((int) x, (int) grid.getHeight(), (int) x, grid.getHeight() - (int) y);
		}

		// draw klein-nishina probability function
		lastX = 0;
		lastY = 0;
		for (int i = 0; i < maxAngle; ++i) {
			double value = XRayTracerSampling.comptonAngleProbability(energyJoule, Math.toRadians(i));
			double x = (i * (grid.getWidth() / 360.f));
			double y = grid.getHeight() - ((value) * (grid.getHeight() / (max)));

			fp.drawLine((int) lastX, (int) lastY, (int) x, (int) y);
			lastX = x;
			lastY = y;
		}

		grid.show("Energy: " + energyJoule / eV + "eV; x-axis: angle[0-360]; y-axis: probability[0-max]");

	}

	public static void main(String[] args) {
		new SingleMaterialMonteCarlo();
	}
}

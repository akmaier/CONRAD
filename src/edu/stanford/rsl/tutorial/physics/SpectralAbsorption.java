/*
 * Copyright (C) 2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.physics;

import ij.gui.Plot;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.physics.absorption.PolychromaticAbsorptionModel;
import edu.stanford.rsl.conrad.physics.absorption.SelectableEnergyMonochromaticAbsorptionModel;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Example class to play and test spectral absorption. Here we generate a spectrum and plot it.
 * Next, we create a monochromatic and a polychromatic absorption model. We investigate both in terms of their absorption behavior and create a plot for the mono- and the polychromatic case.
 * The difference can be looked at, both both are plotted in the same coordinate system. This code is also featured on the CONRAD tutorial website in more detail.
 * @author Andreas Maier
 *
 */
public class SpectralAbsorption {

	public static void main(String [] args){

		// Enable access to the global configuration
		Configuration.loadConfiguration();

		// Fetch some geometry information from the global configuration
		Trajectory trajectory = Configuration.getGlobalConfiguration().getGeometry();
		double detectorArea = trajectory.getPixelDimensionX() * trajectory.getDetectorWidth() * trajectory.getDetectorHeight() * trajectory.getPixelDimensionY();
		double pixelArea = trajectory.getPixelDimensionX() * trajectory.getPixelDimensionY();

		// set spectral parameters
		double min = 10;
		double max = 60;
		double delta = 0.5;
		double peakVoltage = 40;			
		double timeCurrentProduct = 1;	

		// create spectrum
		PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum(min, max, delta, peakVoltage, timeCurrentProduct);
		// get total photon flux in [photons/mm^-2]
		double totalPhotonFlux = spectrum.getTotalPhotonFlux();
		/* input energy in J */
		double inputEnergy = peakVoltage * timeCurrentProduct;
		/* convert to keV */
		inputEnergy *= 6.24150934E15; 
		/* output energy in keV */
		double outputEnergy = detectorArea * totalPhotonFlux * spectrum.getAveragePhotonEnergy();
		/* compute source efficiency, i.e. the amount of energy that is detected compared to the amount of energy that is used for creation of the spectrum */
		double ratio = outputEnergy / inputEnergy;
		// create spectral plot
		Plot plot = new Plot("C-Arm Spectrum with "  + detectorArea * totalPhotonFlux + " total photons and " + pixelArea * totalPhotonFlux + " photons per pixel, avg energy "+ spectrum.getAveragePhotonEnergy() + " [keV] and " +ratio*100+ " % energy efficiency", "Energy [keV]", "Photon Flux [photons/mm^2]", spectrum.getPhotonEnergies(), spectrum.getPhotonFlux());
		plot.show();

		// create absorption / length plots
		// initialize arrays
		int vals = 20;
		double lengths [] = new double [vals];
		double monoAttenuation []= new double[vals];
		double polyAttenuation []= new double[vals];
		// create monochromatic absorption model
		SelectableEnergyMonochromaticAbsorptionModel monochromaticAbsorptionModel = new SelectableEnergyMonochromaticAbsorptionModel();
		monochromaticAbsorptionModel.configure(spectrum.getAveragePhotonEnergy());
		// create polychromatic absorption model
		PolychromaticAbsorptionModel polychromaticAbsorptionModel = new PolychromaticAbsorptionModel();
		polychromaticAbsorptionModel.setInputSpectrum(spectrum);
		try {
			polychromaticAbsorptionModel.configure();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// create list of line intersections.
		// Here we have a line with only one intersection
		// of water
		PhysicalObject segment = new PhysicalObject();
		segment.setMaterial(MaterialsDB.getMaterial("water"));
		segment.setNameString("Water Path");
		ArrayList<PhysicalObject> segments = new ArrayList<PhysicalObject>();
		segments.add(segment);
		// Iterate over the different intersection lengths
		for (int i = 0; i < vals; i++){
			lengths[i] = i*20;
			// create intersection path
			Edge edge = new Edge(new PointND(0), new PointND(lengths[i]));
			segment.setShape(edge);
			// evaluate attenuation models
			monoAttenuation[i]= monochromaticAbsorptionModel.evaluateLineIntegral(segments);
			polyAttenuation[i]= polychromaticAbsorptionModel.evaluateLineIntegral(segments);
		}
		// create plots
		Plot plot2 = new Plot("Monochromatic absorption q, avg energy "+ spectrum.getAveragePhotonEnergy() + " [keV]", "path length in water", "q_mono", lengths, monoAttenuation);
		plot2.show();
		Plot plot3 = new Plot("Polychromatic absorption q, avg energy "+ spectrum.getAveragePhotonEnergy() + " [keV]", "path length in water", "q_poly", lengths, polyAttenuation);
		plot3.show();
	}


}

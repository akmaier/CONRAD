/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.physics.PolychromaticXRaySpectrum;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.gui.Plot;
import ij.plugin.PlugIn;


public class Generate_Spectrum implements PlugIn{

	public Generate_Spectrum (){
		
	}
	
	@Override
	public void run(String arg) {
		try {
			double min = UserUtil.queryDouble("Minimum [keV]", 10);
			double max = UserUtil.queryDouble("Maximum [keV]", 150);
			double delta = UserUtil.queryDouble("Delta [keV]", 0.5);
			double angle = UserUtil.queryDouble("Angle [deg]", 12);
			double peakVoltage = UserUtil.queryDouble("Peak Voltage [kVp]", 120);			
			double timeCurrentProduct = UserUtil.queryDouble("Time Current Product [mAs]", 1);
			
			Trajectory trajectory = Configuration.getGlobalConfiguration().getGeometry();
			double detectorArea = trajectory.getPixelDimensionX() * trajectory.getDetectorWidth() * trajectory.getDetectorHeight() * trajectory.getPixelDimensionY();
			double pixelArea = trajectory.getPixelDimensionX() * trajectory.getPixelDimensionY();
			
			PolychromaticXRaySpectrum spectrum = new PolychromaticXRaySpectrum(min, max, delta, peakVoltage, "W", timeCurrentProduct, trajectory.getSourceToDetectorDistance() / 1000.0, angle, 2.38, 3.06, 2.66, 10.5);
			double totalPhotonFlux = spectrum.getTotalPhotonFlux();
			/* input energy in J */
			double inputEnergy = peakVoltage * timeCurrentProduct;
			/* convert to keV */
			inputEnergy *= 6.24150934E15; 
			/* output energy in keV */
			double outputEnergy = detectorArea * totalPhotonFlux * spectrum.getAveragePhotonEnergy();
			double ratio = outputEnergy / inputEnergy;
			Plot plot = new Plot("C-Arm Spectrum with "  + detectorArea * totalPhotonFlux + " total photons and " + pixelArea * totalPhotonFlux + " photons per pixel, avg energy "+ spectrum.getAveragePhotonEnergy() + " [keV] and " +ratio*100+ " % energy efficiency", "Energy [keV]", "Photon Flux [photons/mm^2]", spectrum.getPhotonEnergies(), spectrum.getPhotonFlux());
			plot.show();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}


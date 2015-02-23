package edu.stanford.rsl.tutorial.fan;

import ij.ImageJ;
import ij.io.Opener;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;

/**
 * This class is used to compare the tutorial fan beam reconstruction pipeline 
 * with the pipeline that is implemented in CONRAD. Therefore, the forward projected 
 * phantoms are imported from a saved file and the central line of each projection
 * is extracted to build a fan beam sinogram.
 * 
 * The tutorial code and the CONRAD code are then compared after several steps, e.g.
 * Cosine weighting, ramp filtering and backprojection.
 * 
 * 
 * Note, that we do not compare any forward projection of the tutorial code and CONRAD!
 * 
 * @author berger
 * 
 */
public class FanBeamComparisonToCONRAD {


	/**
	 * @param args
	 */
	public static void main(String[] args) {

		
		// get geometry details from global configuration ("Conrad.xml")
		Configuration.loadConfiguration();
		Configuration config = Configuration.getGlobalConfiguration();
		Trajectory traj = config.getGeometry();
		// image params
		int imgSzXGU = traj.getReconDimensionX(), // [GU]
				imgSzYGU = traj.getReconDimensionY(); // [GU]
		double pxSzX = traj.getReconVoxelSizes()[0], // [mm]
				pxSzY = traj.getReconVoxelSizes()[1]; // [mm]

		// fan beam bp parameters
		double 	maxT = traj.getDetectorWidth(), 
				deltaT = traj.getPixelDimensionX(), 
				focalLength = traj.getSourceToDetectorDistance(),
				maxBeta = traj.getNumProjectionMatrices()*traj.getAverageAngularIncrement(), 
				deltaBeta = traj.getAverageAngularIncrement(),
				gammaM = Math.atan((maxT/2.0)/focalLength);

		System.out.println(gammaM*180/Math.PI);

		new ImageJ();
		
		// Load projection data and extract sinogram
		Grid2D Sinogram = extractSinogram("C:\\Users\\berger\\Desktop\\WaterCylProjDatShort.zip", traj, false, "ProjectionData");

		// show sinogram
		Sinogram.clone().show("Sinogram");

		// define cosine weights and RamLak kernel and Parker weights
		RamLakKernel ramLak = new RamLakKernel((int) maxT, deltaT);
		CosineFilter cKern = new CosineFilter(focalLength, maxT*deltaT, deltaT);
		ParkerWeights pWeights = new ParkerWeights(focalLength, maxT, deltaT, maxBeta*Math.PI/180, deltaBeta*Math.PI/180);
		// Apply cosine filtering
		for (int theta = 0; theta < Sinogram.getSize()[1]; ++theta) {
			//cKern.applyToGrid(Sinogram.getSubGrid(theta));
		}
		//Sinogram.clone().show("After Cosine Weighting");
		
		// Load cosine weighted projection data and extract sinogram
		Grid2D SinogramCosine = extractSinogram("C:\\Users\\berger\\Desktop\\WaterCylProjDatCosineShort.zip", traj, false, "CosineData");
		
		Grid2D diffCosineFanCONRAD = (Grid2D)NumericPointwiseOperators.subtractedBy(Sinogram,SinogramCosine);
		NumericPointwiseOperators.abs(diffCosineFanCONRAD);
		//diffCosineFanCONRAD.show("Difference Cosine Weighting Tutorial vs. CONRAD");
		
		// Apply Parker Weights
		pWeights.applyToGrid(Sinogram);
		Sinogram.clone().show("After Parker Weighting");
		
		Grid2D SinogramParker = extractSinogram("C:\\Users\\berger\\Desktop\\oneMaskParker.zip", traj, false, "ParkerData");
		SinogramParker.show("CONRAD after Parker");
		
		Grid2D diffParkerFanCONRAD = (Grid2D)NumericPointwiseOperators.subtractedBy(Sinogram,SinogramParker);
		NumericPointwiseOperators.abs(diffParkerFanCONRAD);
		diffParkerFanCONRAD.show("Difference Parker Weighting Tutorial vs. CONRAD");
		
		
		
		// Apply RamLak filter
		for (int theta = 0; theta < Sinogram.getSize()[1]; ++theta) {
			ramLak.applyToGrid(Sinogram.getSubGrid(theta));
		}
		Sinogram.clone().show("After RamLak Filtering Tutorial");

		// Load cosine weighted projection data and extract sinogram
		Grid2D SinogramRamLak = extractSinogram("C:\\Users\\berger\\Desktop\\WaterCylProjDatRamLakShort.zip", traj, false, "RamLakData");
		SinogramRamLak.clone().show("After RamLak Filtering Conrad");
		
		Grid2D diffRamLakFanCONRAD = (Grid2D)NumericPointwiseOperators.subtractedBy(Sinogram,SinogramRamLak);
		NumericPointwiseOperators.abs(diffRamLakFanCONRAD);
		diffRamLakFanCONRAD.show("Difference after RamLak Tutorial vs. CONRAD");
		

		// Do the backprojection
		FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength,
				deltaT, deltaBeta, imgSzXGU, imgSzYGU);
		Grid2D FanbeamRecon = fbp.backprojectPixelDrivenCL(Sinogram);
		FanbeamRecon.show("Reconstruction Result");
		
		
		Opener op = new Opener();		
		Grid3D CONRADRecon = ImageUtil.wrapImagePlus(op.openZip("C:\\Users\\berger\\Desktop\\WaterCylProjDatReconShort.zip"));		
		CONRADRecon.show("CONRAD Reconstruction Result");
		
		Grid2D diffReconFanCONRAD = (Grid2D)NumericPointwiseOperators.subtractedBy(FanbeamRecon, CONRADRecon);
		diffReconFanCONRAD.show("Difference after Recon Tutorial vs. CONRAD");

	}
	
	
	
	static Grid2D extractSinogram(String filename, Trajectory traj, boolean showData, String name)
	{
		Opener op = new Opener();
		Grid3D projData = ImageUtil.wrapImagePlus(op.openZip(filename));
		if (showData)
			projData.show(name);
		
		// extract sinogram out of cone beam projection data (central detector line in v direction)
		Grid2D Sinogram = new Grid2D(projData.getSize()[0], projData.getSize()[2]);
		for (int i=0; i < projData.getSize()[2]; ++i)
		{
			NumericPointwiseOperators.copy(Sinogram.getSubGrid(i), 
					projData.getSubGrid(i).getSubGrid(traj.getDetectorHeight()/2));
		}
		Sinogram.setSpacing(traj.getPixelDimensionX(), traj.getAverageAngularIncrement());
		
		return Sinogram;
		
	}
}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
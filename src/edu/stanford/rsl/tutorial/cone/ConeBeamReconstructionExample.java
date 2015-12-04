package edu.stanford.rsl.tutorial.cone;


import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;

/**
 * Simple example that computes and displays a cone-beam reconstruction.
 * 
 * @author Recopra Seminar Summer 2012
 * 
 */
public class ConeBeamReconstructionExample {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new ImageJ();
		
		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();

		Trajectory geo = conf.getGeometry();
		double focalLength = geo.getSourceToDetectorDistance();
		int maxU_PX = geo.getDetectorWidth();
		int maxV_PX = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		double maxU = (maxU_PX) * deltaU;
		double maxV = (maxV_PX) * deltaV;
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		//Phantom3D test3D = new Sphere3D(imgSizeX, imgSizeY, imgSizeZ);
		Grid3D test3D = new NumericalSheppLogan3D(imgSizeX,
				imgSizeY, imgSizeZ).getNumericalSheppLoganPhantom();
		// Alternate Phantom
		/*
		 * NumericalSheppLogan3D shepp3d = new NumericalSheppLogan3D(imgSizeX,
				imgSizeY, imgSizeZ);
		 */
		Grid3D grid = test3D;
		grid.show("object");

		Grid3D sino;
		ConeBeamProjector cbp =  new ConeBeamProjector();
		try {
			sino = cbp.projectRayDrivenCL(grid);
		} catch (Exception e) {
			System.out.println(e);
			return;
		}
		
		
		ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength, maxU, maxV, deltaU, deltaV);
		RamLakKernel ramK = new RamLakKernel(maxU_PX, deltaU);
		for (int i = 0; i < geo.getProjectionStackSize(); ++i) {
			cbFilter.applyToGrid(sino.getSubGrid(i));
			//ramp
			for (int j = 0;j <maxV_PX; ++j)
				ramK.applyToGrid(sino.getSubGrid(i).getSubGrid(j));
		}
		sino.show("sinoFilt");
			
		ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
		Grid3D recImage = cbbp.backprojectPixelDrivenCL(sino);
		recImage.show("recImage");
		if (true)
			return;
	

	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
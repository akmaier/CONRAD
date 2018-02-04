package edu.stanford.rsl.tutorial.atract;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.motion.estimation.CylinderVolumeMask;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.Phantom3D;
import edu.stanford.rsl.tutorial.phantoms.SimpleCubes3D;
import edu.stanford.rsl.tutorial.phantoms.Sphere3D;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

/**
 * This class is an example implementation to show the usage of the atract filter.
 * The 1D atract implementation is run on a 2D image.
 * The 2D atract implementation is run on a 2D image.
 * the 2D atract implementation is run on a 3D volume.
 * 
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class DisplayAtract {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// sinogram params
		double maxTheta = Math.PI, // [rad]
				deltaTheta = Math.PI / 360, // [rad]
				maxS = 150, // [mm]
				deltaS = 1.0; // [mm]
		// image params
		int imgSzXMM = 150, // [mm]
				imgSzYMM = imgSzXMM; // [mm]
		float pxSzXMM = 1.0f, // [mm]
				pxSzYMM = pxSzXMM; // [mm]

		//float focalLength = 400, maxBeta = (float) Math.PI*2, deltaBeta = maxBeta / 200, maxT = 200, deltaT = 1;

		int phantomType = 3; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
								// 3=DotsGrid
								// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
				imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();

		ParallelProjector2D projector = new ParallelProjector2D(maxTheta, deltaTheta, maxS, deltaS);

		// image object
		Phantom phantom;
		switch (phantomType) {
		case 0:
			phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
			break;
		case 1:
			phantom = new MickeyMouseGrid2D(imgSzXGU, imgSzYGU);
			break;
		case 2:
			phantom = new TestObject1(imgSzXGU, imgSzYGU);
			break;
		case 3:
			phantom = new DotsGrid2D(imgSzXGU, imgSzYGU);
			break;
		default:
			phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
			break;
		}

		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2.0, -(imgSzYGU * phantom.getSpacing()[1]) / 2.0);
		phantom.show();
		Grid2D grid = phantom;

		// create projections
		Grid2D sinoRay = projector.projectRayDriven(grid);
		sinoRay.show("Sino CPU Ray");

		int maxSIndex = (int) (maxS / deltaS + 1);
		int maxThetaIndex = (int) (maxTheta / deltaTheta + 1);
		Kollimator koll = new Kollimator(maxThetaIndex, maxSIndex);

		koll.applyToGrid(sinoRay, 50);
		sinoRay.show("Kolli");

		AtractFilter1D at = new AtractFilter1D();
		at.applyToGrid(sinoRay);

		ParallelBackprojector2D bp = new ParallelBackprojector2D(imgSzXMM, imgSzYMM, pxSzXMM, pxSzYMM);
		Grid2D recon = bp.backprojectRayDriven(sinoRay);

		recon.show("Reconstruction");

		Grid2D sinoRay2 = projector.projectRayDriven(grid);
		sinoRay2.show("Sino CPU Ray");
		RamLakKernel ramLak = new RamLakKernel((int) (maxS / deltaS), deltaS);
		for (int theta = 0; theta < sinoRay2.getSize()[0]; ++theta) {
			ramLak.applyToGrid(sinoRay2.getSubGrid(theta));
		}
		Grid2D recon2 = bp.backprojectRayDriven(sinoRay2);
		recon2.show("Normal Reconstruction");

		Grid2D sinoRay3 = projector.projectRayDriven(grid);
		sinoRay3.show("Sino CPU Ray");

		koll.applyToGrid(sinoRay3, 50);
		sinoRay3.show("Kolli");

		AtractFilter2D at2 = new AtractFilter2D();
		at2.applyToGrid2D(sinoRay3);

		Grid2D recon3 = bp.backprojectRayDriven(sinoRay3);

		recon3.show("Reconstruction");

		@SuppressWarnings("unused")
		double focalLength = 800, maxT = maxS, deltaT = deltaS, gammaM = Math.atan2(maxT / 2.f /*- 0.5*/, focalLength),
				maxBeta = Math.PI * 2, deltaBeta = maxBeta / 180;

		Configuration.loadConfiguration();

		Configuration conf = Configuration.getGlobalConfiguration();

		Trajectory geo = conf.getGeometry();
		int maxU = geo.getDetectorWidth();
		int maxV = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Phantom3D test3D = new SimpleCubes3D(imgSizeX, imgSizeY, imgSizeZ);

		Grid3D grid3 = test3D;
		ConeBeamProjector cbp = new ConeBeamProjector();
		grid3.show("object");
		Grid3D sino = cbp.projectRayDrivenCL(grid3);

		sino.show("sinoCL");
		ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(conf.getGeometry().getSourceToDetectorDistance(),
				conf.getGeometry().getDetectorWidth(), conf.getGeometry().getDetectorHeight(), 1.0, 1.0);

		for (int i = 0; i < conf.getGeometry().getProjectionStackSize(); ++i) {
			cbFilter.applyToGrid(sino.getSubGrid(i));

			float D = (float) conf.getGeometry().getSourceToDetectorDistance();
			NumericPointwiseOperators.multiplyBy(sino.getSubGrid(i),
					(float) (D * D * Math.PI / geo.getNumProjectionMatrices()));
		}
		sino.show("sinoFilt");

		Kollimator koll3 = new Kollimator();

		koll3.applyToGrid(sino, 100, 100);
		sino.show("Kolli");

		AtractFilter2D af = new AtractFilter2D();
		af.applyToGrid2D(sino);

		ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
		Grid3D recImage = cbbp.backprojectPixelDrivenCL(sino);
		CylinderVolumeMask mask = new CylinderVolumeMask(imgSizeX, imgSizeY, imgSizeX / 2, imgSizeY / 2, 20);
		mask.applyToGrid(recImage);
		recImage.show("recImage");

		Grid3D sino2 = cbp.projectRayDrivenCL(grid3);

		koll3.applyToGrid(sino2, 100, 100);
		sino2.show("Kolli");

		RamLakKernel ramK = new RamLakKernel(maxU, deltaU);
		for (int i = 0; i < conf.getGeometry().getProjectionStackSize(); ++i) {
			cbFilter.applyToGrid(sino2.getSubGrid(i));
			for (int j = 0; j < maxV; ++j)
				ramK.applyToGrid(sino2.getSubGrid(i).getSubGrid(j));
			float D = (float) conf.getGeometry().getSourceToDetectorDistance();
			NumericPointwiseOperators.multiplyBy(sino2.getSubGrid(i),
					(float) (D * D * Math.PI / geo.getNumProjectionMatrices()));
		}
		sino2.show("sinoFilt");

		Grid3D recImage2 = cbbp.backprojectPixelDrivenCL(sino2);

		mask.applyToGrid(recImage2);
		recImage2.show("recImage");

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
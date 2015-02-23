package edu.stanford.rsl.tutorial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.Phantom3D;
import edu.stanford.rsl.tutorial.phantoms.Sphere3D;
import edu.stanford.rsl.tutorial.phantoms.TestObject1;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

/**
 * Simple example that computes and displays a reconstruction.
 * 
 * @author Recopra Seminar Summer 2012
 * 
 */
public class DisplayReconstruction {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// sinogram params
		double maxS = 200, // [mm]
		deltaS = 1.0, // [mm]
		maxTheta = Math.PI * 2, // [rad]
		deltaTheta = maxTheta / 200; // [rad]

		// image params
		int imgSzXMM = 100, // [mm]
		imgSzYMM = imgSzXMM; // [mm]
		float pxSzXMM = 1.0f, // [mm]
		pxSzYMM = pxSzXMM; // [mm]
		// fan beam bp parameters
		@SuppressWarnings("unused")
		double focalLength = 600, maxT = maxS, deltaT = deltaS, gammaM = Math
				.atan2(maxT / 2.f /*- 0.5*/, focalLength), maxBeta = Math.PI * 2, deltaBeta = maxBeta / 180;

		int phantomType = 1; // 0 = circle, 1 = MickeyMouse, 2 = TestObject1,
								// 3=DotsGrid
		// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
		imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();
		@SuppressWarnings("unused")
		ParallelProjector2D projector = new ParallelProjector2D(maxTheta,
				deltaTheta, maxS, deltaS);
		@SuppressWarnings("unused")
		ParallelBackprojector2D backprojector = new ParallelBackprojector2D(
				imgSzXGU, imgSzYGU, pxSzXMM, pxSzYMM);

		@SuppressWarnings("unused")
		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(
				focalLength, maxBeta, deltaBeta, maxT, deltaT);
		@SuppressWarnings("unused")
		FanBeamBackprojector2D fanBeamBackprojector = new FanBeamBackprojector2D(
				focalLength, deltaT, deltaBeta, imgSzXGU, imgSzYGU);

		// image object
		// MickeyMouseGrid2D phantom = new MickeyMouseGrid2D(imgSzXGU,
		// imgSzYGU);
		// DotsGrid2D grid = new DotsGrid2D(imgSzXGU, imgSzYGU);
		// UniformCircleGrid2D phantom = new UniformCircleGrid2D(imgSzXGU,
		// imgSzYGU);

		@SuppressWarnings("unused")
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

		Configuration.loadConfiguration();
		
		Configuration conf = Configuration.getGlobalConfiguration();
		
		Trajectory geo = conf.getGeometry();
		focalLength = geo.getSourceToDetectorDistance();
		//int maxU = geo.getDetectorWidth();
		//int maxV = geo.getDetectorHeight();
		int maxU_PX = geo.getDetectorWidth();
		int maxV_PX = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		double maxU = (maxU_PX) * deltaU;
		double maxV = (maxV_PX) * deltaV;
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Phantom3D test3D = new Sphere3D(imgSizeX, imgSizeY, imgSizeZ);
//		test3D.show("Sphere");
		@SuppressWarnings("unused")
		NumericalSheppLogan3D shepp3d = new NumericalSheppLogan3D(imgSizeX,
				imgSizeY, imgSizeZ);

		// phantom.setSpacing(pxSzXMM, pxSzYMM);
		// // origin is given in (negative) world coordinates
		// phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2.0,
		// -(imgSzYGU * phantom.getSpacing()[1]) / 2.0);
		// phantom.show();
		Grid3D grid = test3D;
//		 Grid3D grid =shepp3d.getNumericalSheppLoganPhantom();
		grid.show("object");

		ConeBeamProjector cbp = new ConeBeamProjector();
		//Grid3D sino = cbp.projectPixelDriven(grid);
		Grid3D sino = cbp.projectRayDrivenCL(grid);
		sino.show("sinoCL");
	
		ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength, maxU, maxV, deltaU, deltaV);
		RamLakKernel ramK = new RamLakKernel(maxU_PX, deltaU);
		for (int i = 0; i < conf.getGeometry().getProjectionStackSize(); ++i) {
			cbFilter.applyToGrid(sino.getSubGrid(i));
			//ramp
			for (int j = 0;j <maxV_PX; ++j)
				ramK.applyToGrid(sino.getSubGrid(i).getSubGrid(j));
			float D = (float) conf
					.getGeometry().getSourceToDetectorDistance();
			NumericPointwiseOperators.multiplyBy(sino.getSubGrid(i), (float) (D*D * Math.PI / geo.getNumProjectionMatrices()));
		}
		sino.show("sinoFilt");
		
		ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
		Grid3D recImage = cbbp.backprojectPixelDrivenCL(sino);
		recImage.show("recImage");
		if (true)
			return;
	
		// create projections
		/*
		 * Grid2D sinoRay = projector.projectRayDriven(grid); GridImage
		 * sinoImage = new GridImage(sinoRay); sinoImage.show("Sinogram - Ray");
		 */

		// Grid2D sinoPx = projector.projectRayDriven(grid);
		// sinoPx.show("P Sinogram");

		// Grid2D fanBeamSinoRay = fanBeamProjector.projectRayDrivenCL(grid);
		// fanBeamSinoRay.show("FB CL Sinogram");
		// // if (true)
		// // return;

		// RamLakKernel ramLak = new RamLakKernel((int) (maxS / deltaS),
		// deltaS);
		//ConeBeamCosineFilter cKern = new ConeBeamCosineFilter(focalLength,
		//		maxU, maxV, deltaU, deltaV);
		// // ParkerWeights pWeights = new ParkerWeights(focalLength, maxT,
		// deltaT,
		// // maxBeta, deltaBeta);
		// // pWeights.applyToGrid(fanBeamSinoRay);
		// // pWeights.show();
		//for (int theta = 0; theta < sino.getSize()[0]; ++theta) {
			//cKern.applyToGrid(sino.getSubGrid(theta));
			// // ramLak.applyToGrid(fanBeamSinoRay.getSubGrid(theta));
			// // // ramLak.applyToGrid(sinoPx.getSubGrid(theta));
		//}
		//
		// FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength,
		// deltaT, deltaBeta, imgSzXMM, imgSzYMM);
		// System.out.print("Pixel:");
		// long t0 = System.currentTimeMillis();
		// Grid2D CLPixel = fbp.backprojectPixelDrivenCL(fanBeamSinoRay);
		// System.out.println(System.currentTimeMillis() - t0);
		// t0 = System.currentTimeMillis();
		// Grid2D CLRay = fbp.backprojectRayDrivenCL(fanBeamSinoRay);
		// //fbpCL.show("FB P/CPU Reconstruction");
		// System.out.println("Ray: " + (System.currentTimeMillis() - t0));
		// CLRay.show("Ray");
		// CLPixel.show("Pixel");

	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
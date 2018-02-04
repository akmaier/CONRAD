package edu.stanford.rsl.tutorial.motion.estimation;

import javax.swing.JOptionPane;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceUniformCubicBSpline;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.VTKMeshReader;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.tutorial.motion.compensation.OpenCLCompensatedBackProjector1DCompressionField;
import edu.stanford.rsl.tutorial.motion.compensation.OpenCLCompensatedBackProjectorTPS;

/**
 * Class to run the Motion Estimation.
 * @author Marco Bögel
 *
 */
public class RunMotionEstimation {

	//Small Volume Parameters
	private static final int rSmallX = 128;
	private static final int rSmallY = 128;
	private static final int rSmallZ = 128;
	private static final double vSpaceSmallX = 2.0;
	private static final double vSpaceSmallY = 2.0;
	private static final double vSpaceSmallZ = 2.0;

	//Large Volume Parameters
	private static final int rLargeX = 256;
	private static final int rLargeY = 256;
	private static final int rLargeZ = 512;
	private static final double vSpaceLargeX = 1.0;
	private static final double vSpaceLargeY = 1.0;
	private static final double vSpaceLargeZ = 0.5;

	public static void main(String[] args) throws Exception {

		ImageJ ij = new ImageJ();
		JOptionPane.showMessageDialog(ij, "Load Projection Images");
		String filename = FileUtil.myFileChoose(".zip", false);

		//Set Configuration to small volumes for initial optimization
		Configuration.loadConfiguration();
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		geom.setReconDimensionX(rSmallX);
		geom.setReconDimensionY(rSmallY);
		geom.setReconDimensionZ(rSmallZ);
		geom.setVoxelSpacingX(vSpaceSmallX);
		geom.setVoxelSpacingY(vSpaceSmallY);
		geom.setVoxelSpacingZ(vSpaceSmallZ);
		double xOriginWorld = -(rSmallX - 1.0) / 2.0 * vSpaceSmallX;
		double yOriginWorld = -(rSmallY - 1.0) / 2.0 * vSpaceSmallY;
		double zOriginWorld = -(rSmallZ - 1.0) / 2.0 * vSpaceSmallZ;
		geom.setOriginInPixelsX(General.worldToVoxel(0.0, vSpaceSmallX, xOriginWorld));
		geom.setOriginInPixelsY(General.worldToVoxel(0.0, vSpaceSmallY, yOriginWorld));
		geom.setOriginInPixelsZ(General.worldToVoxel(0.0, vSpaceSmallZ, zOriginWorld));

		Configuration.saveConfiguration();
		Configuration.loadConfiguration();

		//ProjectionLoader
		ProjectionLoader pLoad = new ProjectionLoader();
		pLoad.loadAndFilterImages(filename);

		//Run Initial Optimization
		InitialOptimization initOpti = new InitialOptimization(pLoad);

		float[] initMotionParams = initOpti.optimizeCompressedWithPrior();
		float[][] initMotion = initOpti.getMotionField(initMotionParams, rLargeZ, vSpaceLargeZ,
				-(rLargeZ - 1.0) / 2.0 * vSpaceLargeZ);

		//set Configuration to larger volume
		Configuration.loadConfiguration();
		geom = Configuration.getGlobalConfiguration().getGeometry();
		geom.setReconDimensionX(rLargeX);
		geom.setReconDimensionY(rLargeY);
		geom.setReconDimensionZ(rLargeZ);
		geom.setVoxelSpacingX(vSpaceLargeX);
		geom.setVoxelSpacingY(vSpaceLargeY);
		geom.setVoxelSpacingZ(vSpaceLargeZ);
		xOriginWorld = -(rLargeX - 1.0) / 2.0 * vSpaceLargeX;
		yOriginWorld = -(rLargeY - 1.0) / 2.0 * vSpaceLargeY;
		zOriginWorld = -(rLargeZ - 1.0) / 2.0 * vSpaceLargeZ;
		geom.setOriginInPixelsX(General.worldToVoxel(0.0, vSpaceLargeX, xOriginWorld));
		geom.setOriginInPixelsY(General.worldToVoxel(0.0, vSpaceLargeY, yOriginWorld));
		geom.setOriginInPixelsZ(General.worldToVoxel(0.0, vSpaceLargeZ, zOriginWorld));
		Configuration.saveConfiguration();
		Configuration.loadConfiguration();

		OpenCLCompensatedBackProjector1DCompressionField p = new OpenCLCompensatedBackProjector1DCompressionField();
		ProjectionLoader pLoad2 = new ProjectionLoader();
		JOptionPane.showMessageDialog(ij, "Input Segmented Diaphragm File");
		String file = FileUtil.myFileChoose(".zip", false);
		pLoad2.loadAndFilterImages(file);

		p.loadInputQueue(pLoad2.getProjections());

		Grid3D result = p.reconstructCL(initMotion);
		CylinderVolumeMask mask = new CylinderVolumeMask(result.getSize()[0], result.getSize()[1],
				result.getSize()[0] / 2, result.getSize()[1] / 2, result.getSize()[0] * 0.5);
		mask.applyToGrid(result);
		result.show();

		//Spline
		int sampling = 1;
		VTKMeshReader vRead = new VTKMeshReader();
		JOptionPane.showMessageDialog(ij, "Input Diaphragm Mesh File");
		String vtkname = FileUtil.myFileChoose(".vtk", false);
		vRead.readFile(vtkname);
		EstimateBSplineSurface estimator = new EstimateBSplineSurface(vRead.getPts());
		SurfaceUniformCubicBSpline spline = estimator.estimateUniformCubic(sampling);

		/*
		Grid3D grid = new Grid3D(rLargeX,rLargeY,rLargeZ);
		Configuration c = Configuration.getGlobalConfiguration();
		for (int i = 0; i < grid.getSize()[0]; i++) {
			double u = ((double) i) / (grid.getSize()[0]);
			for (int j = 0; j < grid.getSize()[1]; j++) {
				double v = ((double) j) / (grid.getSize()[1]);
				PointND pt = spline.evaluate(u, v);
		
				if (0 <= -((pt.get(0) + c.getGeometry().getOriginX()) / c
						.getGeometry().getVoxelSpacingX())
						&& 0 <= -((pt.get(1) + c.getGeometry().getOriginY()) / c
								.getGeometry().getVoxelSpacingY())
						&& 0 <= ((pt.get(2) - c.getGeometry().getOriginZ()) / c
								.getGeometry().getVoxelSpacingZ())
						&& -((pt.get(0) + c.getGeometry().getOriginX()) / c
								.getGeometry().getVoxelSpacingX()) < grid
								.getSize()[0]
						&& -((pt.get(1) + c.getGeometry().getOriginY()) / c
								.getGeometry().getVoxelSpacingY()) < grid
								.getSize()[1]
						&& ((pt.get(2) - c.getGeometry().getOriginZ()) / c
								.getGeometry().getVoxelSpacingZ()) < grid
								.getSize()[2])
					grid.setAtIndex(
							(int) -((pt.get(0) + c.getGeometry().getOriginX()) / c
									.getGeometry().getVoxelSpacingX()),
							(int) -((pt.get(1) + c.getGeometry().getOriginY()) / c
									.getGeometry().getVoxelSpacingY()),
							(int) ((pt.get(2) - c.getGeometry().getOriginZ()) / c
									.getGeometry().getVoxelSpacingZ()), 100);
		
			}
		}
		
		grid.show();
		*/

		//Optimize Motionfield
		OptimizeMotionField oMot = new OptimizeMotionField(initMotion, initMotionParams, spline, pLoad);

		OpenCLCompensatedBackProjectorTPS otps = oMot.optimalReconstructor();

		otps.loadInputQueue(pLoad.getProjections());
		Grid3D finalResult = otps.reconstructCL();
		mask.applyToGrid(finalResult);
		finalResult.show();

	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
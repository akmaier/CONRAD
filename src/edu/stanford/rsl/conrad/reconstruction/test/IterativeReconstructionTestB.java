package edu.stanford.rsl.conrad.reconstruction.test;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.reconstruction.iterative.PenalizedLeastSquareART;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class IterativeReconstructionTestB {




	public static void initTrajectory( CircularTrajectory dataTrajectory ){

		SimpleVector rotationAxis = new SimpleVector(0, 0, 1);

		dataTrajectory.setDetectorHeight(128);
		dataTrajectory.setDetectorWidth(256);
		dataTrajectory.setSourceToAxisDistance(400.0);
		dataTrajectory.setSourceToDetectorDistance(800.0);
		dataTrajectory.setReconDimensions(128, 128, 64);
		dataTrajectory.setOriginInPixelsX((dataTrajectory.getReconDimensionX()-1)/2 );
		dataTrajectory.setOriginInPixelsY((dataTrajectory.getReconDimensionY()-1)/2 );
		dataTrajectory.setOriginInPixelsZ((dataTrajectory.getReconDimensionZ()-1)/2 );
		dataTrajectory.setDetectorOffsetU(0);
		dataTrajectory.setDetectorOffsetV(0);
		dataTrajectory.setPixelDimensionX(1.5);
		dataTrajectory.setPixelDimensionY(4.0);
		dataTrajectory.setVoxelSpacingX(1.0);
		dataTrajectory.setVoxelSpacingY(1.0);
		dataTrajectory.setVoxelSpacingZ(2.0);
		dataTrajectory.setAverageAngularIncrement(1.0);
		dataTrajectory.setProjectionStackSize(20);
		dataTrajectory.setDetectorUDirection(Projection.CameraAxisDirection.DETECTORMOTION_PLUS);
		dataTrajectory.setDetectorVDirection(Projection.CameraAxisDirection.ROTATIONAXIS_PLUS);
		dataTrajectory.setTrajectory( 200, 300.0, 1.0 , -0.5, -0.5, Projection.CameraAxisDirection.DETECTORMOTION_PLUS, Projection.CameraAxisDirection.ROTATIONAXIS_PLUS, rotationAxis);
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		float beta = 0.1f;
		float delta = 0.01f; 
		int maxIterantion = 30; 
		
		CircularTrajectory dataTrajectory = new CircularTrajectory();
		initTrajectory(  dataTrajectory );		

		PenalizedLeastSquareART reconSolver =  new PenalizedLeastSquareART( dataTrajectory, maxIterantion, beta, delta );
		//LeastSquaresCG reconSolver =  new LeastSquaresCG( dataTrajectory );



		try {
			reconSolver.initializeTest();
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		try {

			NumericalSheppLogan3D phan = new NumericalSheppLogan3D(dataTrajectory.getReconDimensionX(), dataTrajectory.getReconDimensionY(), dataTrajectory.getReconDimensionZ());

			Grid3D image = phan.getNumericalSheppLoganPhantom();
			VisualizationUtil.showGrid3DZ( image, "Shepp Logan Phantom" ).show();

			Grid3D proj = reconSolver.InitializeProjectionViews();

			reconSolver.forwardproject(proj, image);
			VisualizationUtil.showGrid3DX( proj, "Projection images" ).show();

			reconSolver.initialize(proj);
			reconSolver.iterativeReconstruct();

			VisualizationUtil.showGrid3DZ( reconSolver.getvolumeImage(), "Reconstructed image" ).show();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}


}

/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.conrad.reconstruction.test;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.NumericalSheppLogan3D;
import edu.stanford.rsl.conrad.reconstruction.iterative.DistanceDrivenBasedReconstruction;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


public class IterativeReconstructionTestA {

	public static boolean Debug1 = false;
	public static boolean Debug2 = false;
	public static boolean Debug3 = true;


	public static void initTrajectory( CircularTrajectory dataTrajectory ){

		SimpleVector rotationAxis = new SimpleVector(0, 0, 1);
		
		dataTrajectory.setDetectorHeight(128);
		dataTrajectory.setDetectorWidth(256);
		dataTrajectory.setSourceToAxisDistance(400.0);
		dataTrajectory.setSourceToDetectorDistance(800.0);
		dataTrajectory.setReconDimensions(256, 256, 128);
		dataTrajectory.setOriginInPixelsX((dataTrajectory.getReconDimensionX()-1)/2 );
		dataTrajectory.setOriginInPixelsY((dataTrajectory.getReconDimensionY()-1)/2 );
		dataTrajectory.setOriginInPixelsZ((dataTrajectory.getReconDimensionZ()-1)/2 );
		dataTrajectory.setDetectorOffsetU(0);
		dataTrajectory.setDetectorOffsetV(0);
		dataTrajectory.setPixelDimensionX(2.0);
		dataTrajectory.setPixelDimensionY(4.0);
		dataTrajectory.setVoxelSpacingX(1.0);
		dataTrajectory.setVoxelSpacingY(1.0);
		dataTrajectory.setVoxelSpacingZ(2.0);
		dataTrajectory.setAverageAngularIncrement(1.0);
		dataTrajectory.setProjectionStackSize(20);
		dataTrajectory.setDetectorUDirection(CameraAxisDirection.DETECTORMOTION_PLUS);
		dataTrajectory.setDetectorVDirection(CameraAxisDirection.ROTATIONAXIS_PLUS);
		dataTrajectory.setTrajectory( 200, 300.0, 1.0 , -0.5, -0.5, CameraAxisDirection.DETECTORMOTION_PLUS, CameraAxisDirection.ROTATIONAXIS_PLUS, rotationAxis);
	}


	public static void printSimpleMatrix( SimpleMatrix A ){
		int n = A.getRows();
		int m = A.getCols();

		for (int i = 0; i < n ; i++ ){
			for (int j = 0; j < m ; j++){
				System.out.print( A.getElement(i, j) + "\t");
			}
			System.out.print("\n");
		}
	}

	public static void main(String[] args){

		System.out.println("Hello World!");

		CircularTrajectory dataTrajectory = new CircularTrajectory();
		initTrajectory(  dataTrajectory );

		Projection Pmatrix;
		SimpleVector cameraCenter;
		SimpleMatrix ProjMatrix;

		
		DistanceDrivenBasedReconstruction ddTester =  new DistanceDrivenBasedReconstruction( dataTrajectory );


		if ( Debug1 ){ 
			System.out.println("Hello World Debug1!");
			System.out.println("Detector size: " + dataTrajectory.getDetectorWidth() + " X " + dataTrajectory.getDetectorHeight() );
			System.out.println("Volume size: " + dataTrajectory.getReconDimensionX() + " X " + dataTrajectory.getReconDimensionY() + " X " + dataTrajectory.getReconDimensionZ());
			System.out.println("Detector direction: " + dataTrajectory.getDetectorUDirection() + ", " + dataTrajectory.getDetectorVDirection() );
			for (int p = 0; p < dataTrajectory.getNumProjectionMatrices() ; p ++  ){
				Pmatrix = dataTrajectory.getProjectionMatrix(p);
				cameraCenter =  Pmatrix.computeCameraCenter();
				System.out.println( "Camera center at: " + cameraCenter.getElement(0) + ", " + cameraCenter.getElement(1) );
			}
		}

		
		if (Debug2){

			System.out.println("Hello World Debug2!");

			for (int p = 0; p < dataTrajectory.getNumProjectionMatrices() ; p ++  ){
				Pmatrix = dataTrajectory.getProjectionMatrix(p);
				ProjMatrix =  Pmatrix.computeP();
				System.out.println( "Projection matrix @" + p + ":" );
				printSimpleMatrix( ProjMatrix );
			}

		}
		
		if (Debug3){
			
			
			try {
				ddTester.initializeTest();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			
			try {
				
				NumericalSheppLogan3D phan = new NumericalSheppLogan3D(dataTrajectory.getReconDimensionX(), dataTrajectory.getReconDimensionY(), dataTrajectory.getReconDimensionZ());
				
				Grid3D image = phan.getNumericalSheppLoganPhantom();
				VisualizationUtil.showGrid3DZ( image, "original phantom" ).show();
				
				Grid3D proj = ddTester.InitializeProjectionViews();
				Grid3D vol = ddTester.InitializeVolumeImage();
				
				ddTester.forwardproject(proj, image);
				VisualizationUtil.showGrid3DX( proj, "projection images" ).show();
				
				ddTester.backproject(proj, vol);
				VisualizationUtil.showGrid3DZ( vol, "backprojected image" ).show();
				
				ddTester.printOutGeometry();
				
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	

	}

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
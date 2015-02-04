package edu.stanford.rsl.conrad.reconstruction.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;

public class LeastSquaresCG extends DistanceDrivenBasedReconstruction {

	/**
	 * Model-based iterative reconstruction using least-squares conjugate gradient solver.
	 * by Meng Wu
	 */
	private static final long serialVersionUID = 1L;
	
	static final int maxNumOfIterations = 20;


	public LeastSquaresCG(Trajectory dataTrajectory) {
		super(dataTrajectory);
		// TODO Auto-generated constructor stub
	}


	public void initialize( Grid3D proj){


		maxI = getGeometry().getReconDimensionX();
		maxJ = getGeometry().getReconDimensionY();
		maxK = getGeometry().getReconDimensionZ();
		maxU = getGeometry().getDetectorWidth(); //or it should be projection.getWidth();
		maxV = getGeometry().getDetectorHeight();
		dx = getGeometry().getVoxelSpacingX();
		dy = getGeometry().getVoxelSpacingY();
		dz = getGeometry().getVoxelSpacingZ();
		time = System.currentTimeMillis();

		projectionViews = proj;
		//projectionViews = InitializeProjectionViews();
		//volumeImage = InitializeVolumeImage();

	}


	public void iterativeReconstruct() throws Exception{

		Grid3D r = InitializeProjectionViews();
		Grid3D q = InitializeProjectionViews();
		Grid3D f = InitializeVolumeImage();
		Grid3D d = InitializeVolumeImage();
		Grid3D g_new = InitializeVolumeImage();
		Grid3D g_old = InitializeVolumeImage();

		double gamma;
		double alpha;

		NumericPointwiseOperators.copy(projectionViews, r );

		System.out.print("Model-based iterative reconstruction using least-squares conjugate gradient solver: \n");
		System.out.print("itn   \t||r||  \t\t  x(0) \t\n" );
		System.out.print("---------------------------------------- \n");
		
		for ( int itr = 1; itr <= maxNumOfIterations; itr++ ){

			if ( itr <= 5 || itr%10 == 0 || itr >= maxNumOfIterations - 5 )
			System.out.print( itr + "  \t" + String.format( "%8.3g", NumericPointwiseOperators.dotProduct( r )) +  " \t" 
					+ String.format( "%8.4f", f.getAtIndex(maxJ/2, maxK/2, maxK/2)) + " \n" );
			
			backproject( r, g_new );

			if ( itr == 1 ){
				gamma = 0.0;
				NumericPointwiseOperators.copy( g_new, d );
			}else{
				gamma = NumericPointwiseOperators.dotProduct( g_new ) / NumericPointwiseOperators.dotProduct( g_old );
				NumericPointwiseOperators.multiplyBy(d, (float) gamma);
				NumericPointwiseOperators.addBy(d, g_new);
			}

			forwardproject( q, d );
			
			alpha = NumericPointwiseOperators.dotProduct( g_new, d ) / NumericPointwiseOperators.dotProduct( q );

			NumericGrid tmp = d.clone();
			NumericPointwiseOperators.multiplyBy(tmp, (float) alpha);
			NumericPointwiseOperators.addBy(f, tmp);
			
			tmp = q.clone();
			NumericPointwiseOperators.multiplyBy(tmp, (float) -alpha);
			NumericPointwiseOperators.addBy(r, tmp);

			NumericPointwiseOperators.copy( g_new, g_old);

		}
		volumeImage = f;

	}


}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
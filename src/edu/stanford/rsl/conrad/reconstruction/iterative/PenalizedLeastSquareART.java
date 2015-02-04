package edu.stanford.rsl.conrad.reconstruction.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;

public class PenalizedLeastSquareART extends DistanceDrivenBasedReconstruction {

	/**
	 * Model-based iterative reconstruction using penalized least-squares solver.
	 * by Meng Wu
	 */
	private static final long serialVersionUID = 1L;
	
	int maxNumOfIterations = 30;
	float beta = 0.1f;
	float delta = 0.01f;
	
	public PenalizedLeastSquareART(Trajectory dataTrajectory) {
		super(dataTrajectory);
		// TODO Auto-generated constructor stub
	}
	
	public PenalizedLeastSquareART(Trajectory dataTrajectory, int maxIterations) {
		super(dataTrajectory);
		this.maxNumOfIterations = maxIterations;
	}
	
	public PenalizedLeastSquareART(Trajectory dataTrajectory, int maxIterations, float beta, float delta) {
		super(dataTrajectory);
		this.maxNumOfIterations = maxIterations;
		this.beta = beta;
		this.delta = delta;
	}
	
	huberPenalty pfun = new huberPenalty( beta, delta );

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

	}
	
	public void iterativeReconstruct() throws Exception{
		
		
		Grid3D a = InitializeVolumeImage();
		Grid3D d = InitializeProjectionViews();
		Grid3D e = InitializeVolumeImage();
		
		Grid3D s = InitializeVolumeImage();
		Grid3D t = InitializeVolumeImage();
		
		Grid3D f = InitializeVolumeImage();
		
		//pre-compute denumerator constant
		NumericPointwiseOperators.fill(a, 1.0f);
		forwardproject(d, a);
		backproject(d, a);
		
		forwardproject(d, f); // d = A(f)
		d = (Grid3D)NumericPointwiseOperators.subtractedBy(d, projectionViews);  // d =  A(f) - l
		
		System.out.print("Model-based iterative reconstruction: ");
		System.out.print( "\t beta = " + beta + " delta = " + delta + "\n");
		System.out.print("itn   \t||r||  \t\t  x(0)  \tR(x) \t\n" );
		System.out.print("------------------------------------------------ \n");
		
		for ( int itr = 1; itr <= maxNumOfIterations; itr++ ){

			backproject( d, e ); //e = At(d) 
			
			pfun.huberDerivative(f, s);
			//PointwiseOperators.fill(s, 0.0f);
			NumericPointwiseOperators.addBy(s, e); // s = e + s
			
			
			pfun.huberCurvature(f, t);
			//PointwiseOperators.fill(t, 0.0f);
			NumericPointwiseOperators.addBy(t, a); // t = t + a
			
			NumericPointwiseOperators.divideBy(e, t); // e = s ./ t
			
			//update here
			NumericPointwiseOperators.subtractBy(f, e); // f = f - e
			
			NumericPointwiseOperators.removeNegative(f);	
			
			forwardproject(d, f); // d = A(f)
			NumericPointwiseOperators.subtractBy(d, projectionViews); // d = A(f) - l
			
			if ( itr <= 5 || itr%10 == 0 || itr >= maxNumOfIterations - 5 )
			System.out.print( itr + "  \t" + String.format( "%8.3g", NumericPointwiseOperators.dotProduct( d )) +  " \t" 
					+ String.format( "%8.4f", f.getAtIndex(maxJ/2, maxK/2, maxK/2))  +  " \t"
					+ String.format( "%8.4f", pfun.huber(f)) +  " \n" );
			
		}

		volumeImage = f;

	}
	
	

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
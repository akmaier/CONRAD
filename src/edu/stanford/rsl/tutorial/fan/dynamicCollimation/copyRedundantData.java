package edu.stanford.rsl.tutorial.fan.dynamicCollimation;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.BinaryWeights;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;



public class copyRedundantData extends Grid2D {
	
	private final double focalLength;
	private final double maxT;
	private final double deltaT, deltax, dLambda;
	private final int maxTIndex, maxLambdaIndex;
	

	public copyRedundantData(final double focalLength, final double maxT,
			final double deltaT, double maxLambda, double dLambda) {
		// Call constructor from superclass
		super((int) Math.round(maxT / deltaT), (int)(Math.round(maxLambda / dLambda)) + 1);
		// Initialize parameters
		this.focalLength = focalLength;
		this.maxT = maxT;
		this.deltaT = deltaT;
		this.dLambda = dLambda;
		this.maxLambdaIndex = (int)(Math.round(maxLambda / dLambda)) + 1;
		this.maxTIndex = (int) Math.round(maxT / deltaT);

		this.deltax = maxLambda - Math.PI; 
		
		// Correct for scaling due to varying angle lambda
		NumericPointwiseOperators.multiplyBy(this, (float)( maxLambda / (Math.PI)));
		
	}
	
	
	private void createFullSinogram(Grid2D OneSidedSinogram)
	{
		double lambda, delta;
		
		// iterate over the detector elements
		for (int t = 0; t < maxTIndex; ++t) {
			// compute delta of the current ray (detector element)
			delta = Math.atan((t * deltaT - maxT / 2.d + 0.5*deltaT) / focalLength);

			// iterate over the projection angles
			for (int b = 0; b < maxLambdaIndex; ++b) {
				// compute the current lambda angle
				lambda = b * dLambda;
				
				// First case: Handles values for redundancies at the end of the scan
				// Copy values from redundancies at the beginning of the scan
				if (lambda >= ( Math.PI + 2*delta) && lambda <= (Math.PI + deltax) + 1e-12)
				{
					//double delta2 = -1.0*delta;
					double lambda2 = -2.0*delta - Math.PI + lambda;
					double b2 = lambda2 / dLambda;
					//b2 = (b2 < 0) ? (0.0) : b2;
					int t2 = maxTIndex - t -1;//(int) Math.round(delta2 / deltaT);
					
					OneSidedSinogram.setAtIndex(maxTIndex - t - 1, b, InterpolationOperators.interpolateLinear(OneSidedSinogram, b2, maxTIndex - t2 - 1));
				}
			}
		}
	}
	
	
	
	public void applyToGrid(Grid2D OneSidedSinogram) {
		createFullSinogram(OneSidedSinogram);
	}

	
	public static void main (String [] args){
		//fan beam bp parameters
		
		double maxT = 100;
		double deltaT = 1.d;
		// set focal length according to the fan angle
		double focalLength = (maxT/2.0-0.5)/Math.tan(20.0*Math.PI/180.0);
		
		Phantom ph = new SheppLogan(64);
		new ImageJ();
		
		int startBeta = 100;
		int endBeta = 260;
		
		Grid3D g  = new Grid3D((int)maxT, 133, endBeta-startBeta +1, false);
		
		for (int i = startBeta; i < endBeta+1; ++i)
		{
			double maxBeta =  (double)(i+1) * Math.PI * 2.0 / 360.0;
			double deltaBeta = maxBeta / 132;
			
			FanBeamProjector2D fbp_forward = new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
			Grid2D halfSino = fbp_forward.projectRayDriven(ph);
			
			BinaryWeights BW = new BinaryWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
			//BW.show();
			BW.applyToGrid(halfSino);
			//halfSino.show();
			
			copyRedundantData p = new copyRedundantData(focalLength, maxT, deltaT, maxBeta, deltaBeta);
			p.applyToGrid(halfSino);
			Grid2D dummy = new Grid2D(halfSino);
			g.setSubGrid(i-startBeta, dummy);
			//g.setSliceLabel("MaxBeta: " + Double.toString(maxBeta*180/Math.PI), i+1 - startBeta);
		}
		
		g.show();
		
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
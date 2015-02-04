package edu.stanford.rsl.tutorial.fan.redundancy;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;



public class BinaryWeights_Normal extends Grid2D {

	private final double focalLength;
	private final double maxT;
	private final double deltaT, dLambda;
	private final int maxTIndex, maxLambdaIndex;
	private Grid2D binaryMask;


	public BinaryWeights_Normal(final double focalLength, final double maxT,
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
		//this.gammaM = (double) (Math.atan(((maxT-deltaT) / 2.d)/ focalLength) );
		

		binaryMask = new Grid2D(this);
		createWeights();
		createMask();
		
		// Correct for scaling due to varying angle lambda
		NumericPointwiseOperators.multiplyBy(this, (float)( maxLambda / (Math.PI)));
	}
	
	
	private void createWeights()
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
				
				// Default weight is 1
				this.setAtIndex(maxTIndex - t -1, b, 1.0f);
				
				// First case: Handles weights for redundancies at the end of the scan
				// Set weights to zero (Simulates the collimator)
				if (lambda >= ( Math.PI + 2*delta))
				{
					this.setAtIndex(maxTIndex - t -1, b, 0.f );
				}
			}
		}
	}
	
	
	private void createMask()
	{
		
			binaryMask = new Grid2D(this);
			// iterate over the projection angles
			for (int b = 0; b < maxLambdaIndex; ++b) {
				// iterate over the detector elements
				for (int t = 0; t < maxTIndex; ++t) {
					if (t < maxTIndex -1)
					{
						if (this.getAtIndex(t, b) != this.getAtIndex(t+1, b))
						{
							binaryMask.setAtIndex(t, b, 0.f);
						}
					}
			}
		}
	}

	public Grid2D getBinaryMask()
	{
		return binaryMask;
	}
	
	public void applyToGrid(Grid2D sino) {
		NumericPointwiseOperators.multiplyBy(sino, this);
	}

	public static void main (String [] args){
		//fan beam bp parameters
		
		double maxT = 400;
		double deltaT = 1.d;
		// set focal length according to the fan angle
		double focalLength = (maxT/2.0-0.5)/Math.tan(30.0*Math.PI/180.0);
		//double gammaM =Math.atan2(maxT / 2.0 - 0.5, focalLength);	
		new ImageJ();
		
		Grid3D g  = new Grid3D((int)maxT, 181, 360, false);
		for (int i = 0; i < 360; ++i)
		{
			double maxBeta =  (double)(i+1) * Math.PI * 2.0 / 360.0;
			double deltaBeta = maxBeta / 180;
			
			BinaryWeights_Normal p = new BinaryWeights_Normal(focalLength, maxT, deltaT, maxBeta, deltaBeta);
			g.setSubGrid(i, p);
			//g.setSliceLabel("MaxBeta: " + Double.toString(maxBeta*180/Math.PI), i+1);
		}
		
		g.show();
		
		double maxBeta =  Math.PI;
		double deltaBeta = maxBeta / 180;
		
		BinaryWeights_Normal p = new BinaryWeights_Normal(focalLength, maxT, deltaT, maxBeta, deltaBeta);
		p.show();
		p.getBinaryMask().show();
		
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
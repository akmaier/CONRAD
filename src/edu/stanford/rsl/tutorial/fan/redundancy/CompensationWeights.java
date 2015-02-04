package edu.stanford.rsl.tutorial.fan.redundancy;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;


public class CompensationWeights extends Grid2D {

	private final double deltax;
	private final int maxTIndex, maxLambdaIndex;



	public double EtaFunction(double delta, double deltax, double lambda)
	{
		double val = Math.sin((Math.PI/2) * (Math.PI + deltax - lambda) / (deltax - 2 * delta));
		val = val * val;
		return val;
	}


	public double ZetaFunction(double delta, double deltax, double lambda)
	{
		double val = Math.sin((Math.PI/2) * (lambda) / (deltax + 2 * delta));
		val = val * val;
		return val;
	}


	public CompensationWeights(final double focalLength, final double maxT,
			final double deltaT, double maxLambda, double dLambda) {
		// Call constructor from superclass
		super((int) Math.round(maxT / deltaT), (int)(Math.round(maxLambda / dLambda)) + 1);
		
		// Initialize parameters
		this.maxLambdaIndex = (int)(Math.round(maxLambda / dLambda)) + 1;
		this.maxTIndex = (int) Math.round(maxT / deltaT);
		//double gammaM = (double) (Math.atan((maxT / 2.d)/ focalLength) );
		this.deltax = maxLambda - Math.PI; 
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
				
				// Check if zeta and eta are evaluable, otherwise assign a 1 to the weights
				if (deltax + 2 * delta == 0)
					continue;				
				if (deltax - 2 * delta == 0)
					continue;
				
				// implement the conditions as described in the paper
				// First case: Handles weights for redundancies at the end of the scan
				if (lambda >= (Math.PI + 2*delta) && lambda <= (Math.PI + deltax) + 1e-12)
				{
					this.setAtIndex(maxTIndex - t -1, b, (float) (this.EtaFunction(delta, deltax, lambda)) );
				}
				// Second case: Handles weights for missing data at the end of the scan
				if (lambda >= (Math.PI + 2*deltax - 2*delta) && lambda <= (Math.PI + deltax) + 1e-12)
				{
					this.setAtIndex(maxTIndex - t -1, b, (float) (2 - this.EtaFunction(delta, deltax, lambda)) );
				}
				// Third case: Handles weights for redundancies at the beginning of the scan
				if (lambda >= 0 && lambda <= (2*delta + deltax) + 1e-12) 
				{
					this.setAtIndex(maxTIndex - t -1, b, (float) this.ZetaFunction(delta, deltax, lambda));
				}
				// Fourth case: Handles weights for missing data at the beginning of the scan
				if (lambda >= 0 && lambda <= (-1.0 * (2*delta + deltax)) + 1e-12)
				{
					this.setAtIndex(maxTIndex - t -1, b, (float) (2 - this.ZetaFunction(delta, deltax, lambda)) );
				}

			}
		}

		
		// Correct for scaling due to varying angle lambda
		NumericPointwiseOperators.multiplyBy(this, (float)( maxLambda / (Math.PI)));
		
		
		// Low pass filtering of first and last 15 columns to avoid step functions
		// Sigma of gaussian filter decreases towards the middle
		int columnsToFilter = 15;
		double maxSigma = 20;
		double minSigma = 0.1;
		double sigma = maxSigma;
		// Low pass filtering of the first 5 rows + the last 5 rows
		for (int b=0; b<maxLambdaIndex; ++b)
		{			
			// decrease sigma the further we come to the center of the object
			if ( b < columnsToFilter)
				sigma = maxSigma - ((double)b)*(maxSigma-minSigma)/((double)columnsToFilter-1.0);
			if (b >= maxLambdaIndex-columnsToFilter)
				sigma = minSigma + ((double)(b-maxLambdaIndex+columnsToFilter))*(maxSigma-minSigma)/((double)columnsToFilter-1.0);
			
			if ( b < columnsToFilter || b >= maxLambdaIndex-columnsToFilter)
			{
			double column[] = new double[maxTIndex];
			for(int t=0; t < maxTIndex; ++t)
				column[t]=this.getAtIndex(t, b);
			
			column = DoubleArrayUtil.gaussianFilter(column, sigma);
			
			for(int t=0; t < maxTIndex; ++t)
				this.setAtIndex(t, b, (float)column[t]);
			}
		}
		
		
	}

	
	public void applyToGrid(Grid2D sino) {
		NumericPointwiseOperators.multiplyBy(sino, this);
	}

	
	public static void main (String [] args){
		//fan beam bp parameters
		
		double maxT = 400;
		double deltaT = 1.d;
		// set focal length according to the fan angle
		double focalLength = (maxT/2.0-0.5)/Math.tan(10.0*Math.PI/180.0);
		//double gammaM =Math.atan2(maxT / 2.f - 0.5, focalLength) * 2.0;	
		new ImageJ();
		
		
		Grid3D g  = new Grid3D((int)maxT, 181, 360, false);
		
		for (int i = 0; i < 360; ++i)
		{
			double maxBeta =  (double)(i+1) * Math.PI * 2.0 / 360.0;
			double deltaBeta = maxBeta / 180;
			
			CompensationWeights p = new CompensationWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
			g.setSubGrid(i, p);
			//g.setSliceLabel("MaxBeta: " + Double.toString(maxBeta*180/Math.PI), i+1);
		}
		
		g.show();
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
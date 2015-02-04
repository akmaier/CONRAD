package edu.stanford.rsl.tutorial.fan.redundancy;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;

// Variable according to Slaney 
// Note: Parker weights are not defined for super-short scans (maxBeta < pi + 2*gammaM) !! Use Silver or Compensation weights instead!

public class ParkerWeights extends Grid2D {

	private final double gammaM;
	private final int maxTIndex, maxBetaIndex;

	public ParkerWeights(final double focalLength, final double maxT,
			final double deltaT, double maxBeta, double deltaBeta) {
		// Call constructor from superclass
		super((int) Math.round(maxT / deltaT), (int)Math.round(maxBeta / deltaBeta) + 1);
		
		// Initialize parameters
		this.maxBetaIndex = (int)(Math.round(maxBeta / deltaBeta)) + 1;
		this.maxTIndex = (int) Math.round(maxT / deltaT);
		this.gammaM = (double) (Math.atan((maxT / 2.d)/ focalLength) );
		double beta, alpha;

		// iterate over the detector elements
		for (int t = 0; t < maxTIndex; ++t) {
			// compute alpha of the current ray (detector element)
			alpha = Math.atan((t * deltaT - maxT / 2.d + 0.5*deltaT) / focalLength);
			
			// iterate over the projection angles
			for (int b = 0; b < maxBetaIndex; ++b) {
				beta = b * deltaBeta;
				
				// Shift weights such that they are centered (Important for maxBeta < pi + 2 * gammaM)
					beta += (Math.PI+2*gammaM-maxBeta)/2.0;
				
				// Adjust beta if out of range [0, 2*pi]
				if (beta < 0) {
					continue;
				}
				if (beta > Math.PI *2.d) {
					continue;
				}

				// implement the conditions as described in Parker's paper
				if (beta <= 2 * (gammaM - alpha)) {
					double tmp = beta * Math.PI / 4.d / (gammaM - alpha);
					float val = (float) Math.pow(Math.sin(tmp), 2.d);
					
					if (Double.isNaN(val)){
						continue;
					}
					this.setAtIndex(t, b , val);

				} else if (beta < Math.PI - 2.d * alpha) {
					this.setAtIndex(t, b , 1);
				}
				else if (beta <= (Math.PI + 2.d * gammaM) + 1e-12) {
					double tmp = (Math.PI / 4.d) * ( (Math.PI + 2.d*gammaM - beta) / (gammaM + alpha) );
					float val = (float) Math.pow(Math.sin(tmp), 2.d);
					if (Double.isNaN(val)){
						continue;
					}
					this.setAtIndex(t, b , val);
				}
			}
		}
		
		// Correct for scaling due to varying angle
		NumericPointwiseOperators.multiplyBy(this, (float)( maxBeta / (Math.PI)));

	}

	public void applyToGrid(Grid2D sino) {
		NumericPointwiseOperators.multiplyBy(sino, this);
	}

	public static void main (String [] args){
		//fan beam bp parameters

		double maxT = 285;
		//double gammaM =Math.atan2(maxT / 2.f - 0.5, focalLength) * 2.0;
		double fan = 10.0*Math.PI/180.0;
		double focalLength = (maxT/2.0-0.5)/Math.tan(fan);
		new ImageJ();
		double deltaT = 1.d;
		
		Grid3D g = new Grid3D((int)maxT, 181, 360, false);
		for (int i = 0; i < 360; ++i)
		{
			double maxBeta =  (double)(i+1) * Math.PI * 2.0 / 360.0;
			double deltaBeta = maxBeta / 180;
			
			ParkerWeights p = new ParkerWeights(focalLength, maxT, deltaT, maxBeta, deltaBeta);
			//g.setSliceLabel("MaxBeta: " + Double.toString(maxBeta*180/Math.PI), i+1);
			g.setSubGrid(i, p);
			
		}
		
		g.show();

	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
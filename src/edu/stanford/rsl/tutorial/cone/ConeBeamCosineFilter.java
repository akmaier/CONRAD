package edu.stanford.rsl.tutorial.cone;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;

public final class ConeBeamCosineFilter {

	private final double focalLength;
	private final Grid2D weightingKernel;
	private double maxU;
	private double maxV;
	private int maxUIndex;
	private int maxVIndex;
	private double deltaU;
	private double deltaV;

	public ConeBeamCosineFilter(double focalLength, double maxU, double maxV, double deltaU, double deltaV){
		this.focalLength = focalLength;
		this.maxU = maxU;
		this.maxV = maxV;
		this.maxUIndex = (int) (maxU/deltaU);
		this.maxVIndex = (int) (maxV/deltaV);
		this.deltaU = deltaU;
		this.deltaV = deltaV;
		this.weightingKernel = getWeightingFactor();
	}

	private Grid2D getWeightingFactor() {
		final Grid2D weightingKernelTmp = new Grid2D(new float[maxVIndex*maxUIndex], maxVIndex, maxUIndex);
		for(int vIdx=0; vIdx<maxVIndex; ++vIdx){
			double v = vIdx * deltaV - maxV/2.f;
			for(int uIdx=0; uIdx<maxUIndex; ++uIdx){
				double u = uIdx * deltaU - maxU/2.f;
				weightingKernelTmp.setAtIndex(vIdx, uIdx, (float)(focalLength/
						Math.sqrt(focalLength*focalLength + u*u + v*v)));
			}
		}
		return weightingKernelTmp;
	}

	public void applyToGrid(Grid2D input) {
		NumericPointwiseOperators.multiplyBy(input, weightingKernel);
	}
	
	public void show(){
		weightingKernel.show("ConeBeamCosineFilter weightedKernel");
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.fan;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.tutorial.DerivativeKernel;
import edu.stanford.rsl.tutorial.HilbertKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.MickeyMouseGrid2D;

public final class CosineFilter {

	private final double focalLength;
	private final double maxT;
	private final double deltaT;
	private final int maxTIndex;
	private final Grid1D weightingKernel;

	public CosineFilter(final double focalLength, final double maxT, final double deltaT){
		this.focalLength = focalLength;
		this.maxT = maxT;
		this.deltaT = deltaT;
		this.maxTIndex = (int) Math.round(maxT/deltaT);
		this.weightingKernel = getWeightingFactor();
		
	}

	private Grid1D getWeightingFactor() {
		final Grid1D weightingKernelTmp = new Grid1D(maxTIndex);
		for(int i=0; i<maxTIndex; ++i){
			double t = i * deltaT - maxT/2 + 0.5 * deltaT;
			weightingKernelTmp.setAtIndex(i, (float)(focalLength/
					Math.sqrt(focalLength*focalLength+t*t)));
		}
		return weightingKernelTmp;
	}

	public void applyToGrid(Grid1D input) {
		NumericPointwiseOperators.multiplyBy(input, weightingKernel);
//		for(int i=0; i<input.getSize()[0]; ++i){
//			input.multiplyAtIndex(i, weightingKernel.getAtIndex(i));
//		}
	}


















	public final static void main(String[] args) {

		// sinogram params
		double maxTheta = Math.PI * 2,
				deltaTheta = maxTheta / 456,
				maxS = 412,
				deltaS = 2.34;
		// image params
		int imgSzXMM = 212,            // [mm]
				imgSzYMM = imgSzXMM;       // [mm]
		float pxSzXMM = 1.0f,          // [mm]
				pxSzYMM = pxSzXMM;         // [mm]

		// size in grid units
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
				imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		new ImageJ();
		ParallelProjector2D projector = new ParallelProjector2D(maxTheta, deltaTheta, maxS, deltaS);

		// image object
		//UniformCircleGrid2D phantom = new UniformCircleGrid2D(imgSzXGU, imgSzYGU);
		//TestObject1 phantom = new TestObject1(imgSzXGU, imgSzYGU);
		MickeyMouseGrid2D phantom = new MickeyMouseGrid2D(imgSzXGU, imgSzYGU);
		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(phantom.getSize()[0]*phantom.getSpacing()[0]) / 2.0, -(phantom.getSize()[1]*phantom.getSpacing()[1]) / 2.0);
		Grid2D grid = phantom;

		// create projections
		Grid2D sinoRay = projector.projectRayDriven(grid);
		

		//		RamLakKernel ramLak = new RamLakKernel((int) (maxS / deltaS), deltaS);
		DerivativeKernel dKern = new DerivativeKernel();
		HilbertKernel hKern = new HilbertKernel(deltaS);

		for (int theta = 0; theta < sinoRay.getSize()[0]; ++theta) {
			//			ramLak.applyToGrid(sinoRay.getSubGrid(theta));
			dKern.applyToGrid(sinoRay.getSubGrid(theta));
			hKern.applyToGrid(sinoRay.getSubGrid(theta));
		}
		ParallelBackprojector2D bp = new ParallelBackprojector2D(imgSzXGU, imgSzYGU, pxSzXMM, pxSzYMM);
		Grid2D res = bp.backprojectPixelDriven(sinoRay);
		VisualizationUtil.showGrid2D(res, "backprojection");
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
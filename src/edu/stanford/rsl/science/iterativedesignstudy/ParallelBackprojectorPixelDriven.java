package edu.stanford.rsl.science.iterativedesignstudy;


import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class ParallelBackprojectorPixelDriven implements Backprojector{
	final double samplingRate = 3.d;
	int maxThetaIndex;
	int maxSIndex; // image dimensions [GU]
	double maxTheta;
	double deltaTheta; // detector angles [rad]
	double maxS;
	double deltaS;             // detector (pixel) size [mm]

	public ParallelBackprojectorPixelDriven(NumericGrid sino){
		this.maxThetaIndex = sino.getSize()[1];
		this.deltaTheta = sino.getSpacing()[1];
		this.maxTheta = (maxThetaIndex -1) * deltaTheta;
		this.maxSIndex = sino.getSize()[0];
		this.deltaS = sino.getSpacing()[0];
		this.maxS = (maxSIndex-1) * deltaS;
	}
	
	
	public NumericGrid backproject(NumericGrid sino, NumericGrid grid) {
		return this.backprojectPixelDriven((Grid2D)sino,(Grid2D) grid);
	}

	
	public NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) {
		return this.backprojectPixelDriven((Grid1D) projection ,(Grid2D) grid, index);
	}
	
	public Grid2D backprojectPixelDriven(Grid2D sino, Grid2D grid) {
//		this.initSinogramParams(sino);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);
		// loop over the projection angles
		for (int i = 0; i < maxThetaIndex; i++) {
			// compute actual value for theta
			double theta = deltaTheta * i;
			// precompute sine and cosines for faster computation
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);
			// get detector direction vector
			SimpleVector dirDetector = new SimpleVector(sinTheta,cosTheta);
			// loops over the image grid

			for (int x = 0; x < grid.getSize()[0]; x++) {
				for (int y = 0; y < grid.getSize()[1]; y++) {	

					// compute world coordinate of current pixel
					double[] w = grid.indexToPhysical(x, y);
					// wrap into vector
					SimpleVector pixel = new SimpleVector(w[0], w[1]);
					//  project pixel onto detector
					double s = SimpleOperators.multiplyInnerProd(pixel, dirDetector);
					// compute detector element index from world coordinates
					s += maxS/2; // [mm]
					s /= deltaS; // [GU]
					// get detector grid
					Grid1D subgrid = sino.getSubGrid(i);
					// check detector bounds, continue if out of array
					if (subgrid.getSize()[0] <= s + 1
							||  s < 0)
						continue;
					// get interpolated value
					float val = InterpolationOperators.interpolateLinear(subgrid, s);
					// sum value to sinogram
					grid.addAtIndex(y, x, val);
				}

			}
		}
		// apply correct scaling
		NumericPointwiseOperators.divideBy(grid, (float) (maxThetaIndex / Math.PI));
		
		return grid;

	}

	public Grid2D backprojectPixelDriven(Grid1D projection, Grid2D grid, int e) {
		NumericPointwiseOperators.fill(grid, 0);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);

		// loop over the projection angles

		// compute actual value for theta
//		double theta = deltaTheta * e/180.0*Math.PI;
		
		// Create flower artifact
		double theta = deltaTheta * e;
		
		// precompute sine and cosines for faster computation
		double cosTheta = Math.cos(theta);
		double sinTheta = Math.sin(theta);
		// get detector direction vector
		SimpleVector dirDetector = new SimpleVector(sinTheta,cosTheta);
		// loops over the image grid

		for (int x = 0; x < grid.getSize()[0]; x++) {
			for (int y = 0; y < grid.getSize()[1]; y++) {	

				// compute world coordinate of current pixel
				double[] w = grid.indexToPhysical(x, y);
				// wrap into vector
				SimpleVector pixel = new SimpleVector(w[0], w[1]);
				//  project pixel onto detector
				double s = SimpleOperators.multiplyInnerProd(pixel, dirDetector);
				// compute detector element index from world coordinates
				s += maxS/2; // [mm]
				s /= deltaS; // [GU]
				// get detector grid
				// check detector bounds, continue if out of array
				if (projection.getSize()[0] <= s + 1
						||  s < 0)
					continue;
				// get interpolated value
				float val = InterpolationOperators.interpolateLinear(projection, s);
				// sum value to sinogram
				grid.addAtIndex(y, x, val);

			}
		}
		// apply correct scaling
		NumericPointwiseOperators.divideBy(grid, (float) (maxThetaIndex / Math.PI));
		return grid;
	}
}

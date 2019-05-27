package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class ParallelProjectorPixelDriven extends Projector{


	double maxTheta, deltaTheta, maxS, deltaS;				// [mm]
	int maxThetaIndex, maxSIndex;
	
	public ParallelProjectorPixelDriven(double maxTheta, double deltaTheta, double maxS,double deltaS) {
		this.maxS = maxS;
		this.maxTheta = maxTheta;
		this.deltaS = deltaS;
		this.deltaTheta = deltaTheta;
		this.maxSIndex = (int) (maxS / deltaS + 1);
		this.maxThetaIndex = (int) (maxTheta / deltaTheta + 1);
	}
	
	public NumericGrid project(NumericGrid grid,NumericGrid sino){
		return this.projectPixelDriven((Grid2D) grid,(Grid2D)sino); 
	}
	
	public Grid2D projectPixelDriven(Grid2D grid, Grid2D sino) {

		for (int i = 0; i < maxThetaIndex; i++) {
			double theta = deltaTheta * i;
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);

			SimpleVector dirDetector = new SimpleVector(cosTheta, sinTheta);

			// loop over all grid points
			// x,y are in the grid coordinate system
			// wx,wy are in the world coordinate system
			for (int x = 0; x < grid.getSize()[0]; x++) {
				for (int y = 0; y < grid.getSize()[1]; y++) {
					float val = (float) (grid.getAtIndex(x, y)/deltaS);
					val *= grid.getSpacing()[0] * grid.getSpacing()[1]; // assuming isometric pixels
					double[] w = grid.indexToPhysical(x, y);
					double wx = w[0], wy = w[1]; // convenience
					SimpleVector pixel = new SimpleVector(wx, wy);
					double s = SimpleOperators.multiplyInnerProd(pixel,
							dirDetector);
					s += maxS/2;
					s /= deltaS;
					
					Grid1D subgrid = sino.getSubGrid(i);
					if (subgrid.getSize()[0] <= s + 1
							||  s < 0)
						continue;

					InterpolationOperators
							.addInterpolateLinear(subgrid, s, val);
				}
			}
		}

		return sino;
	}

	public NumericGrid project(NumericGrid grid, NumericGrid sino, int index) {
		return null;
	}


}

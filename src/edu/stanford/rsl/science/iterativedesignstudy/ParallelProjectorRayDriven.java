package edu.stanford.rsl.science.iterativedesignstudy;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class ParallelProjectorRayDriven extends Projector{
	double maxTheta, deltaTheta,maxS, deltaS;				// [mm]
	int maxThetaIndex, maxSIndex;

	/**
	 * Sampling of projections is defined in the constructor.
	 * 
	 * @param maxTheta the angular range in radians
	 * @param deltaTheta the angular step size in radians
	 * @param maxS the detector size in [mm]
	 * @param deltaS the detector element size in [mm]
	 */
	public ParallelProjectorRayDriven(double maxTheta, double deltaTheta, double maxS,double deltaS) {
		this.maxS = maxS;
		this.maxTheta = maxTheta;
		this.deltaS = deltaS;
		this.deltaTheta = deltaTheta;
		this.maxSIndex = (int) (maxS / deltaS + 1);
		this.maxThetaIndex = (int) (maxTheta / deltaTheta + 1);
	}
	
		public Grid2D projectRayDriven(Grid2D grid,Grid2D sino) {
			final double samplingRate = 3.d; // # of samples per pixel
			// set up image bounding box in WC
			Translation trans = new Translation(
					-(grid.getSize()[0] * grid.getSpacing()[0])/2, -(grid.getSize()[1] * grid.getSpacing()[1])/2, -1
				);
			Transform inverse = trans.inverse();

			Box b = new Box((grid.getSize()[0] * grid.getSpacing()[0]), (grid.getSize()[1] * grid.getSpacing()[1]), 2);
			b.applyTransform(trans);

			for(int e=0; e<maxThetaIndex; ++e){
				// compute theta [rad] and angular functions.
				double theta = deltaTheta * e;
				double cosTheta = Math.cos(theta);
				double sinTheta = Math.sin(theta);

				for (int i = 0; i < maxSIndex; ++i) {
					// compute s, the distance from the detector edge in WC [mm]
					double s = deltaS * i - maxS / 2;
					// compute two points on the line through s and theta
					// We use PointND for Points in 3D space and SimpleVector for directions.
					PointND p1 = new PointND(s * cosTheta, s * sinTheta, .0d);
					PointND p2 = new PointND(-sinTheta + (s * cosTheta),
							(s * sinTheta) + cosTheta, .0d);
					// set up line equation
					StraightLine line = new StraightLine(p1, p2);
					// compute intersections between bounding box and intersection line.
					ArrayList<PointND> points = b.intersect(line);

					// only if we have intersections
					if (2 != points.size()){
						if(points.size() == 0) {
							line.getDirection().multiplyBy(-1.d);
							points = b.intersect(line);
						}
						if(points.size() == 0)
							continue;
					}

					PointND start = points.get(0); // [mm]
					PointND end = points.get(1);   // [mm]

					// get the normalized increment
					SimpleVector increment = new SimpleVector(
							end.getAbstractVector());
					increment.subtract(start.getAbstractVector());
					double distance = increment.normL2();
					increment.divideBy(distance * samplingRate);

					double sum = .0;
					start = inverse.transform(start);

					// compute the integral along the line.
					for (double t = 0.0; t < distance * samplingRate; ++t) {
						PointND current = new PointND(start);
						current.getAbstractVector().add(increment.multipliedBy(t));

						double x = current.get(0) / grid.getSpacing()[0],
								y = current.get(1) / grid.getSpacing()[1];

						if (grid.getSize()[0] <= x + 1
								|| grid.getSize()[1] <= y + 1
								|| x < 0 || y < 0)
							continue;

						sum += InterpolationOperators.interpolateLinear(grid, x, y);
					}

					// normalize by the number of interpolation points
					sum /= samplingRate;
					// write integral value into the sinogram.
					sino.setAtIndex(i, e, (float)sum);
				}
			}
			return sino;
		}
	public NumericGrid project(NumericGrid grid,NumericGrid sino){
		return this.projectRayDriven((Grid2D) grid,(Grid2D)sino); 
	}



	public NumericGrid project(NumericGrid grid, NumericGrid sino, int index) {
		return this.projectRayDriven1D((Grid2D) grid,(Grid1D)sino, index); 
	}
	
	
		public Grid1D projectRayDriven1D(Grid2D grid,Grid1D sino, int e) {
			final double samplingRate = 3.d; // # of samples per pixel

			// set up image bounding box in WC
			Translation trans = new Translation(
					-(grid.getSize()[0] * grid.getSpacing()[0])/2, -(grid.getSize()[1] * grid.getSpacing()[1])/2, -1
				);
			Transform inverse = trans.inverse();

			Box b = new Box((grid.getSize()[0] * grid.getSpacing()[0]), (grid.getSize()[1] * grid.getSpacing()[1]), 2);
			b.applyTransform(trans);

//			for(int e=0; e<maxThetaIndex; ++e){
				// compute theta [rad] and angular functions.
				double theta =  deltaTheta * e;
				double cosTheta = Math.cos(theta);
				double sinTheta = Math.sin(theta);

				for (int i = 0; i < maxSIndex; ++i) {
					// compute s, the distance from the detector edge in WC [mm]
					double s = deltaS * i - maxS / 2;
					// compute two points on the line through s and theta
					// We use PointND for Points in 3D space and SimpleVector for directions.
					PointND p1 = new PointND(s * cosTheta, s * sinTheta, .0d);
					PointND p2 = new PointND(-sinTheta + (s * cosTheta),
							(s * sinTheta) + cosTheta, .0d);
					// set up line equation
					StraightLine line = new StraightLine(p1, p2);
					// compute intersections between bounding box and intersection line.
					ArrayList<PointND> points = b.intersect(line);

					// only if we have intersections
					if (2 != points.size()){
						if(points.size() == 0) {
							line.getDirection().multiplyBy(-1.d);
							points = b.intersect(line);
						}
						if(points.size() == 0)
							continue;
					}

					PointND start = points.get(0); // [mm]
					PointND end = points.get(1);   // [mm]

					// get the normalized increment
					SimpleVector increment = new SimpleVector(
							end.getAbstractVector());
					increment.subtract(start.getAbstractVector());
					double distance = increment.normL2();
					increment.divideBy(distance * samplingRate);

					double sum = .0;
					start = inverse.transform(start);

					// compute the integral along the line.
					for (double t = 0.0; t < distance * samplingRate; ++t) {
						PointND current = new PointND(start);
						current.getAbstractVector().add(increment.multipliedBy(t));

						double x = current.get(0) / grid.getSpacing()[0],
								y = current.get(1) / grid.getSpacing()[1];

						if (grid.getSize()[0] <= x + 1
								|| grid.getSize()[1] <= y + 1
								|| x < 0 || y < 0)
							continue;

						sum += InterpolationOperators.interpolateLinear(grid, x, y);
					}

					// normalize by the number of interpolation points
					sum /= samplingRate;
					// write integral value into the sinogram.
					sino.setAtIndex(i, (float)sum);
				}
//			}
				//sino.show("sino");
			return sino;
			
		}		
}	

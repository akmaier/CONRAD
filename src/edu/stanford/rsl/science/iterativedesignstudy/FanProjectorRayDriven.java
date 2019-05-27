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

public class FanProjectorRayDriven extends Projector{
	final double samplingRate = 3.d;

	private double focalLength, deltaBeta, maxT, deltaT;

	int maxTIndex, maxBetaIndex;

	/** 
	 * Creates a new instance of a fan-beam projector
	 * 
	 * @param focalLength the focal length
	 * @param maxBeta the maximal rotation angle
	 * @param deltaBeta the step size between source positions
	 * @param maxT the length of the detector array
	 * @param deltaT the size of one detector element
	 */
	public FanProjectorRayDriven(double focalLength,double maxBeta, double deltaBeta,double maxT, double deltaT) {

		this.focalLength = focalLength;
		this.maxT = maxT;
		this.deltaBeta = deltaBeta;
		this.deltaT = deltaT;
		this.maxBetaIndex = (int) (maxBeta / deltaBeta + 1);
		this.maxTIndex = (int) (maxT / deltaT);
	}
	public NumericGrid project(NumericGrid grid, NumericGrid sino) {
		return this.projectRayDriven((Grid2D)grid,(Grid2D)sino);
	}

	public NumericGrid project(NumericGrid grid, NumericGrid sino, int index) {
		return this.projectRayDriven1D((Grid2D) grid,(Grid1D) sino, index);
	}

	public Grid1D projectRayDriven1D(Grid2D grid,Grid1D sino, int i) {		
		// create translation to the grid origin
		Translation trans = new Translation(-grid.getSize()[0] / 2.0,-grid.getSize()[1] / 2.0, -1);
		// build the inverse translation
		Transform inverse = trans.inverse();

		// set up image bounding box and translate to origin
		Box b = new Box(grid.getSize()[0], grid.getSize()[1], 2);
		b.applyTransform(trans);

		// compute the current rotation angle and its sine and cosine
		double beta = deltaBeta * i;
		double cosBeta = Math.cos(beta);
		double sinBeta = Math.sin(beta);

		// compute source position
		PointND a = new PointND(focalLength * cosBeta, focalLength
				* sinBeta, 0.d);
		// compute end point of detector
		PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f
				* cosBeta, 0.d);

		// create an unit vector that points along the detector
		SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
		dirDetector.normalizeL2();

		// iterate over the detector elements
		for (int t = 0; t < maxTIndex; t++) {
			// calculate current bin position
			// the detector elements' position are centered
			double stepsDirection = 0.5f * deltaT + t * deltaT;
			PointND p = new PointND(p0);
			p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));

			// create a straight line between detector bin and source
			StraightLine line = new StraightLine(a, p);

			// find the line's intersection with the box
			ArrayList<PointND> points = b.intersect(line);

			// if we have two intersections build the integral 
			// otherwise continue with the next bin
			if (2 != points.size()) {
				if (points.size() == 0) {
					line.getDirection().multiplyBy(-1.d);
					points = b.intersect(line);
					if (points.size() == 0)
						continue;
				} else {
					continue; // last possibility:
					// a) it is only one intersection point (exactly one of the boundary vertices) or
					// b) it are infinitely many intersection points (along one of the box boundaries).
					// c) our code is wrong
				}

			}

			// Extract intersections
			PointND start = points.get(0);
			PointND end = points.get(1);

			// get the normalized increment
			SimpleVector increment = new SimpleVector(end.getAbstractVector());
			increment.subtract(start.getAbstractVector());
			double distance = increment.normL2();
			increment.divideBy(distance * samplingRate);

			double sum = .0;
			start = inverse.transform(start);

			// compute the integral along the line
			for (double tLine = 0.0; tLine < distance * samplingRate; ++tLine) {
				PointND current = new PointND(start);
				current.getAbstractVector().add(increment.multipliedBy(tLine));
				if (grid.getSize()[0] <= current.get(0) + 1
						|| grid.getSize()[1] <= current.get(1) + 1
						|| current.get(0) < 0 || current.get(1) < 0)
					continue;

				sum += InterpolationOperators.interpolateLinear(grid,
						current.get(0), current.get(1));
			}

			// normalize by the number of interpolation points
			sum /= samplingRate;
			// write integral value into the sinogram.
			sino.setAtIndex(t, (float) sum);
		}
		return sino;
	}

	public Grid2D projectRayDriven(Grid2D grid, Grid2D sino) {
		// create translation to the grid origin
		Translation trans = new Translation(-grid.getSize()[0] / 2.0,
				-grid.getSize()[1] / 2.0, -1);
		// build the inverse translation
		Transform inverse = trans.inverse();
		
		// set up image bounding box and translate to origin
		Box b = new Box(grid.getSize()[0], grid.getSize()[1], 2);
		b.applyTransform(trans);

		// iterate over the rotation angle
		for (int i = 0; i < maxBetaIndex; i++) {
			// compute the current rotation angle and its sine and cosine
			double beta = deltaBeta * i;
			double cosBeta = Math.cos(beta);
			double sinBeta = Math.sin(beta);
//			System.out.println(beta / Math.PI * 180);
			// compute source position
			PointND a = new PointND(focalLength * cosBeta, focalLength
					* sinBeta, 0.d);
			// compute end point of detector
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f
					* cosBeta, 0.d);

			// create an unit vector that points along the detector
			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			dirDetector.normalizeL2();

			// iterate over the detector elements
			for (int t = 0; t < maxTIndex; t++) {
				// calculate current bin position
				// the detector elements' position are centered
				double stepsDirection = 0.5f * deltaT + t * deltaT;
				PointND p = new PointND(p0);
				p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));
				
				// create a straight line between detector bin and source
				StraightLine line = new StraightLine(a, p);
				
				// find the line's intersection with the box
				ArrayList<PointND> points = b.intersect(line);
				
				// if we have two intersections build the integral 
				// otherwise continue with the next bin
				if (2 != points.size()) {
					if (points.size() == 0) {
						line.getDirection().multiplyBy(-1.d);
						points = b.intersect(line);
						if (points.size() == 0)
							continue;
					} else {
						continue; // last possibility:
						 // a) it is only one intersection point (exactly one of the boundary vertices) or
						 // b) it are infinitely many intersection points (along one of the box boundaries).
						 // c) our code is wrong
					}
					
				}

				// Extract intersections
				PointND start = points.get(0);
				PointND end = points.get(1);

				// get the normalized increment
				SimpleVector increment = new SimpleVector(end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				double distance = increment.normL2();
				increment.divideBy(distance * samplingRate);

				double sum = .0;
				start = inverse.transform(start);				
				// compute the integral along the line
				for (double tLine = 0.0; tLine < distance * samplingRate; ++tLine) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(tLine));
					if (grid.getSize()[0] <= current.get(0) + 1
							|| grid.getSize()[1] <= current.get(1) + 1
							|| current.get(0) < 0 || current.get(1) < 0)
						continue;

					sum += InterpolationOperators.interpolateLinear(grid,
							current.get(0), current.get(1));
				}

				// normalize by the number of interpolation points
				sum /= samplingRate;
				// write integral value into the sinogram.
				sino.setAtIndex(t, i, (float) sum);
			}
		}
		return sino;
	}


}

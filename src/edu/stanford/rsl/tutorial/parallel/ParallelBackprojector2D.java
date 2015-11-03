package edu.stanford.rsl.tutorial.parallel;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Implementation of a simple parallel backprojector. In order to create the
 * image, the angular range and the angular sampling and the detector element
 * size and detector element sampling has to be defined. We show both, a ray
 * driven and a pixel driven backprojector.
 * 
 * See L. Zeng. "Medical Image Reconstruction: A Conceptual tutorial". 2009, page 8
 * 
 * @author Recopra Seminar Summer 2012
 * 
 */

/** just testing something */
public class ParallelBackprojector2D {
	final double samplingRate = 3.d;

	int maxThetaIndex, maxSIndex; // image dimensions [GU]
	double maxTheta, deltaTheta,  // detector angles [rad]
	maxS, deltaS;             // detector (pixel) size [mm]
	int imgSizeX, imgSizeY;       // image dimensions [GU]
	float pxSzXMM, pxSzYMM;       // pixel size [mm]

	/**
	 * Sampling of projections is defined in the constructor.
	 * 
	 * @param imageSizeX
	 * @param imageSizeY
	 * @param pxSzXMM
	 * @param pxSzYMM	 */
	public ParallelBackprojector2D(int imageSizeX, int imageSizeY, float pxSzXMM, float pxSzYMM){
		this.imgSizeX = imageSizeX;
		this.imgSizeY = imageSizeY;
		this.pxSzXMM = pxSzXMM;
		this.pxSzYMM = pxSzYMM;
	}

	void initSinogramParams(Grid2D sino) {
		this.maxThetaIndex = sino.getSize()[1];
		this.deltaTheta = sino.getSpacing()[1];
		this.maxTheta = (maxThetaIndex -1) * deltaTheta;

		this.maxSIndex = sino.getSize()[0];
		this.deltaS = sino.getSpacing()[0];
		this.maxS = (maxSIndex-1) * deltaS;
	}

	/**
	 * The ray driven solution.
	 * 
	 * @param sino
	 *            the sinogram
	 * @return the image
	 */
	public Grid2D backprojectRayDriven(Grid2D sino) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);
		grid.setSpacing(this.pxSzXMM, this.pxSzYMM);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0]) / 2.0, -(grid.getSize()[1]*grid.getSpacing()[1]) / 2.0);

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
				SimpleVector increment = new SimpleVector(end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				double distance = increment.normL2();
				increment.divideBy(distance * samplingRate);

				float val = sino.getAtIndex(i, e);
				start = inverse.transform(start);

				// compute the integral along the line.
				for (double t = 0.0; t < distance * samplingRate; ++t) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(t));

					double x = current.get(0) / grid.getSpacing()[0],
							y = current.get(1) / grid.getSpacing()[1];

					if (x + 1 > grid.getSize()[0] 
							|| y + 1 >= grid.getSize()[1]
									|| x < 0 || y < 0)
						continue;

					InterpolationOperators.addInterpolateLinear(grid, x, y, val);
				}

			}
		}
		//		float correctionFactor = (float) ((float) samplingRate * maxThetaIndex / maxTheta * Math.PI);
		float normalizationFactor = (float) ((float) samplingRate * maxThetaIndex / deltaS / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);
		return grid;
	}

	/**
	 * The pixel driven solution.
	 * 
	 * @param sino
	 *            the sinogram
	 * @return the image
	 */
	public Grid2D backprojectPixelDriven(Grid2D sino) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);
		grid.setSpacing(pxSzXMM, pxSzYMM);
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
					grid.addAtIndex(x, y, val);
				}

			}
		}
		// apply correct scaling
		NumericPointwiseOperators.divideBy(grid, (float) (maxThetaIndex / Math.PI));
		return grid;
	}

	/**
	 * The pixel driven solution. for one angle
	 * 
	 * @param sino
	 *            the sinogram
	 * @return the image
	 */
	public Grid2D backprojectPixelDriven(Grid2D sino, int i) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);
		grid.setSpacing(pxSzXMM, pxSzYMM);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);

		// compute actual value for theta
		double theta = deltaTheta * i;
		// precompute sine and cosines for faster computation
		double cosTheta = Math.cos(theta);
		double sinTheta = Math.sin(theta);
		// get detector direction vector
		SimpleVector dirDetector = new SimpleVector(cosTheta,sinTheta);
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
				grid.addAtIndex(x, y, val);
			}

		}

		// apply correct scaling
		NumericPointwiseOperators.divideBy(grid, (float) (maxThetaIndex / Math.PI));
		return grid;
	}

	/**
	 * The pixel driven solution with median filter on the backprojection valeus.
	 * 
	 * @param sino
	 *            the sinogram
	 * @return the image
	 */
	public Grid2D backprojectPixelDrivenPercentile(Grid2D sino, double percentile) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);
		grid.setSpacing(pxSzXMM, pxSzYMM);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);

		SimpleVector [] directionVectors = new SimpleVector[maxThetaIndex];

		int start = (int) (maxThetaIndex *percentile);
		int end = maxThetaIndex -start;
		float range = end - start;

		// precompute the direction vectors for faster computation
		for (int i = 0; i < maxThetaIndex; i++) {
			// compute actual value for theta
			double theta = deltaTheta * i;
			// precompute sine and cosines for faster computation
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);
			// get detector direction vector
			directionVectors[i] = new SimpleVector(sinTheta,cosTheta);
		}

		// loop over the reconstruction grid
		for (int x = 0; x < grid.getSize()[0]; x++) {
			for (int y = 0; y < grid.getSize()[1]; y++) {
				// loop over the projection angles
				double [] components = new double[maxThetaIndex];

				for (int i = 0; i < maxThetaIndex; i++) {
					// compute world coordinate of current pixel
					double[] w = grid.indexToPhysical(x, y);
					// wrap into vector
					SimpleVector pixel = new SimpleVector(w[0], w[1]);
					//  project pixel onto detector
					double s = SimpleOperators.multiplyInnerProd(pixel, directionVectors[i]);
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
					components[i] = InterpolationOperators.interpolateLinear(subgrid, s);
					// sum value to sinogram

				}
				// sort array
				Arrays.sort(components);
				// backproject median value
				for (int i = start; i <= end; i++) {
					grid.addAtIndex(x, y, (float)components[i]);
				}
			}
		}
		// apply correct scaling
		NumericPointwiseOperators.divideBy(grid, (float) (range / Math.PI));
		return grid;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
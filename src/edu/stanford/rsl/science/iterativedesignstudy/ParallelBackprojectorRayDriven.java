package edu.stanford.rsl.science.iterativedesignstudy;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class ParallelBackprojectorRayDriven implements Backprojector{
	final double samplingRate = 3.d;
	int maxThetaIndex;
	int maxSIndex; // image dimensions [GU]
	double maxTheta;
	double deltaTheta; // detector angles [rad]
	double maxS;
	double deltaS; // detector (pixel) size [mm]


	private void initSinogramParams(Grid2D sino) {
		this.maxThetaIndex = sino.getSize()[1];
		this.deltaTheta = sino.getSpacing()[1];
		this.maxTheta = (maxThetaIndex - 1) * deltaTheta;
		this.maxSIndex = sino.getSize()[0];
		this.deltaS = sino.getSpacing()[0];
		this.maxS = (maxSIndex - 1) * deltaS;
	}

	public NumericGrid backproject(NumericGrid sino, NumericGrid grid) {
		return this.backprojectRayDriven((Grid2D) sino, (Grid2D) grid);
	}

	public NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) {
		return null;
	}

	public Grid2D backprojectRayDriven(Grid2D sino, Grid2D grid) {
		this.initSinogramParams(sino);
		grid.setOrigin(-((grid.getSize()[0]-1) * grid.getSpacing()[0]) / 2.0,
				-((grid.getSize()[1]-1) * grid.getSpacing()[1]) / 2.0);

		// set up image bounding box in WC
		Translation trans = new Translation(
				-(grid.getSize()[0] * grid.getSpacing()[0]) / 2,
				-(grid.getSize()[1] * grid.getSpacing()[1]) / 2, -1);
		Transform inverse = trans.inverse();

		Box b = new Box(((grid.getSize()[0]-1) * grid.getSpacing()[0]),
				((grid.getSize()[1]-1) * grid.getSpacing()[1]), 2);
		b.applyTransform(trans);

		for (int e = 0; e < maxThetaIndex; ++e) {
			// compute theta [rad] and angular functions.
			double theta = deltaTheta * e;
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);

			for (int i = 0; i < maxSIndex; ++i) {
				// compute s, the distance from the detector edge in WC [mm]
				double s = deltaS * i - maxS / 2;
				// compute two points on the line through s and theta
				// We use PointND for Points in 3D space and SimpleVector for
				// directions.
				PointND p1 = new PointND(s * cosTheta, s * sinTheta, .0d);
				PointND p2 = new PointND(-sinTheta + (s * cosTheta),
						(s * sinTheta) + cosTheta, .0d);
				// set up line equation
				StraightLine line = new StraightLine(p1, p2);
				// compute intersections between bounding box and intersection
				// line.
				ArrayList<PointND> points = b.intersect(line);

				// only if we have intersections
				if (2 != points.size()) {
					if (points.size() == 0) {
						line.getDirection().multiplyBy(-1.d);
						points = b.intersect(line);
					}
					if (points.size() == 0)
						continue;
				}

				PointND start = points.get(0); // [mm]
				PointND end = points.get(1); // [mm]

				// get the normalized increment
				SimpleVector increment = new SimpleVector(
						end.getAbstractVector());
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

					if (x + 1 > grid.getSize()[0] || y + 1 >= grid.getSize()[1]
							|| x < 0 || y < 0)
						continue;

					InterpolationOperators.addInterpolateLinear(grid, x, y, val);
				}

			}
		}
		// float correctionFactor = (float) ((float) samplingRate * maxThetaIndex / maxTheta * Math.PI);
		float normalizationFactor = (float) ((float) samplingRate * maxThetaIndex / deltaS / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);
		return grid;
	}
}

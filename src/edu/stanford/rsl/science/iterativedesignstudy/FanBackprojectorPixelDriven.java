package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class FanBackprojectorPixelDriven implements Backprojector {
	final double samplingRate = 3.d;
	private float focalLength;
	private float deltaT;
	private float deltaBeta;
	private float maxT;
	private int maxTIndex, maxBetaIndex;
	private int imgSizeX, imgSizeY;

	public FanBackprojectorPixelDriven(double focalLength, int imageSizeX, int imageSizeY, Grid2D sino) {
		this.focalLength = (float) focalLength;
		this.imgSizeX = imageSizeX;
		this.imgSizeY = imageSizeY;

		this.maxTIndex = sino.getSize()[0];
		this.deltaT = (float) sino.getSpacing()[0];
		this.maxT = (float) (maxTIndex * deltaT);
		this.maxBetaIndex = sino.getSize()[1];
		this.deltaBeta = (float) sino.getSpacing()[1];
	}


	public NumericGrid backproject(NumericGrid sino, NumericGrid grid) {
		return this.backproject((Grid2D) sino, (Grid2D) grid);
	}

	public NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) {
		return this.backprojectPixelDriven1D((Grid1D) projection, (Grid2D) grid, index);
	}

	public Grid2D backprojectPixelDriven(Grid2D sino, Grid2D grid) {
		for (int b = 0; b < maxBetaIndex; ++b) {
			// compute beta [rad] and angular functions.
			float beta = (float) (deltaBeta * b);
			float cosBeta = (float) Math.cos(beta);
			float sinBeta = (float) Math.sin(beta);

			PointND a = new PointND(focalLength * cosBeta, focalLength
					* sinBeta, 0.d);
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f
					* cosBeta, 0.d);

			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			StraightLine detectorLine = new StraightLine(p0, dirDetector);

			Grid1D subSino = sino.getSubGrid(b);

			for (int x = 0; x < imgSizeX; ++x) {
				float wx = (x - imgSizeX / 2.0f);

				for (int y = 0; y < imgSizeY; ++y) {
					float wy = (y - imgSizeY / 2.0f);

					// compute two points on the line through t and beta
					// We use PointND for points in 3D space and SimpleVector
					// for directions.
					final PointND reconstructionPointWorld = new PointND(wx,
							wy, 0.d);

					final StraightLine projectionLine = new StraightLine(a,
							reconstructionPointWorld);
					final PointND detectorPixel = projectionLine
							.intersect(detectorLine);

					float valtemp;
					if (detectorPixel != null) {
						final SimpleVector p = SimpleOperators.subtract(
								detectorPixel.getAbstractVector(),
								p0.getAbstractVector());
						double len = p.normL2();
						double t = (len - 0.5d) / deltaT;

						if (subSino.getSize()[0] <= t + 1 || t < 0)
							continue;

						float val = InterpolationOperators.interpolateLinear(
								subSino, t);

						// DistanceWeighting
						float radius = (float) reconstructionPointWorld
								.getAbstractVector().normL2();
						float phi = (float) ((Math.PI / 2) + Math.atan2(
								reconstructionPointWorld.get(1),
								reconstructionPointWorld.get(0)));
						float dWeight = (float) ((focalLength + radius
								* Math.sin(beta - phi)) / focalLength);
						valtemp = (float) (val / (dWeight * dWeight));

					} else {
						valtemp = 0.f;
					}

					grid.addAtIndex(x, y, valtemp);
				}
			}

		}

		float normalizationFactor = (float) (maxBetaIndex / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);

		return grid;
	}

	public Grid2D backprojectPixelDriven1D(Grid1D projection, Grid2D grid, int index) {
		// compute beta [rad] and angular functions.
		float beta = (float) (deltaBeta * index);
		float cosBeta = (float) Math.cos(beta);
		float sinBeta = (float) Math.sin(beta);

		PointND a = new PointND(focalLength * cosBeta, focalLength * sinBeta,
				0.d);
		PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta,
				0.d);

		SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
		StraightLine detectorLine = new StraightLine(p0, dirDetector);

		Grid1D subSino = projection;

		for (int x = 0; x < imgSizeX; ++x) {
			float wx = (x - imgSizeX / 2.0f);

			for (int y = 0; y < imgSizeY; ++y) {
				float wy = (y - imgSizeY / 2.0f);

				// compute two points on the line through t and beta
				// We use PointND for points in 3D space and SimpleVector for
				// directions.
				final PointND reconstructionPointWorld = new PointND(wx, wy,
						0.d);

				final StraightLine projectionLine = new StraightLine(a,
						reconstructionPointWorld);
				final PointND detectorPixel = projectionLine
						.intersect(detectorLine);

				float valtemp;
				if (detectorPixel != null) {
					final SimpleVector p = SimpleOperators.subtract(
							detectorPixel.getAbstractVector(),
							p0.getAbstractVector());
					double len = p.normL2();
					double t = (len - 0.5d) / deltaT;

					if (subSino.getSize()[0] <= t + 1 || t < 0)
						continue;

					float val = InterpolationOperators.interpolateLinear(
							subSino, t);

					// DistanceWeighting
					float radius = (float) reconstructionPointWorld
							.getAbstractVector().normL2();
					float phi = (float) ((Math.PI / 2) + Math.atan2(
							reconstructionPointWorld.get(1),
							reconstructionPointWorld.get(0)));
					float dWeight = (float) ((focalLength + radius
							* Math.sin(beta - phi)) / focalLength);
					valtemp = (float) (val / (dWeight * dWeight));
				} else {
					valtemp = 0.f;
				}

				grid.addAtIndex(x, y, valtemp);
			}
		}

		float normalizationFactor = (float) (maxBetaIndex / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);

		return grid;
	}

}

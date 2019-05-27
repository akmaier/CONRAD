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

public class FanBackprojectorRayDriven implements Backprojector{
	final double samplingRate = 3.d;
	private float focalLength;
	private float deltaT;
	private float deltaBeta;
	private float maxT;
	private int maxTIndex, maxBetaIndex;

	public FanBackprojectorRayDriven(double focalLength, Grid2D sino) {
		this.focalLength = (float) focalLength;

		this.maxTIndex = sino.getSize()[0];
		this.deltaT = (float) sino.getSpacing()[0];
		this.maxT = (float) (maxTIndex * deltaT);
		this.maxBetaIndex = sino.getSize()[1];
		this.deltaBeta = (float) sino.getSpacing()[1];
	}

	public NumericGrid backproject(NumericGrid sino, NumericGrid grid) {
		return this.backprojectRayDriven((Grid2D) sino, (Grid2D) grid);
	}

	@Override
	public NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) {
		// TODO Auto-generated method stub
		return null;
	}
	public Grid2D backprojectRayDriven(Grid2D sino, Grid2D grid) {
//		this.initSinogramParams(sino);
		// set up image bounding box in WC
		Translation trans = new Translation(-grid.getSize()[0]/2.0, -grid.getSize()[1]/2.0, -1);
		// set up the inverse transform
		Transform inverse = trans.inverse();

		Box b = new Box(grid.getSize()[0], grid.getSize()[1], 2);
		b.applyTransform(trans);

		// iterate over the projection angles
		for(int e=0; e<maxBetaIndex; ++e){
			// compute beta [rad] and angular functions.
			float beta = (float) (deltaBeta * e);
			float cosBeta = (float) Math.cos(beta);
			float sinBeta = (float) Math.sin(beta);
			
			// We use PointND for points in 3D space and SimpleVector for directions.
			// compute source location and the beginning of the detector
			PointND a = new PointND(focalLength * cosBeta, focalLength * sinBeta, 0.d);
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta, 0.d);

			// compute the normalized vector along the detector
			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			dirDetector.normalizeL2();

			// iterate over the detector bins
			for (int i = 0; i < maxTIndex; ++i) {
				
				// compute bin position
				float stepsDirection = (float) (0.5f * deltaT + i * deltaT);
				PointND p = new PointND(p0);
				p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));
				// set up line equation
				StraightLine line = new StraightLine(a, p);
				// compute intersections between bounding box and intersection line.
				ArrayList<PointND> points = b.intersect(line);

				// only if we have intersections
				if (2 != points.size()){
					if(points.size() == 0) {
						line.getDirection().multiplyBy(-1.f);
						points = b.intersect(line);
					}
					if(points.size() == 0)
						continue;
				}

				PointND start = points.get(0);
				PointND end = points.get(1);

				// get the normalized increment
				SimpleVector increment = new SimpleVector(end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				float distance = (float) increment.normL2();
				increment.divideBy(distance * samplingRate);

				
				float val = sino.getAtIndex(i, e);
				start = inverse.transform(start);

				// compute the integral along the line.
				for (float t = 0.0f; t < distance * samplingRate; ++t) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(t));
					if (grid.getSize()[0] <= current.get(0) + 1
							|| grid.getSize()[1] <= current.get(1) + 1
							|| current.get(0) < 0 || current.get(1) < 0)
						continue;

					//DistanceWeighting
					PointND currentTrans = new PointND(current);
					currentTrans.applyTransform(trans);
					float radius = (float) currentTrans.getAbstractVector().normL2();
					float phi = (float) (Math.PI/2 + Math.atan2(currentTrans.get(1), currentTrans.get(0)));
					float dWeight = (float) ((focalLength  + radius*Math.sin(beta - phi))/focalLength);
					float valtemp = (float) (val / (dWeight*dWeight));

					//grid.addAtIndex((int)Math.floor(current.get(0)), (int)Math.floor(current.get(1)), valtemp);
					InterpolationOperators.addInterpolateLinear(grid, current.get(0), current.get(1), valtemp);
				}
			}
		}
		float normalizationFactor = (float) ((float) samplingRate * maxBetaIndex / deltaT / Math.PI);
		NumericPointwiseOperators.multiplyBy(grid, normalizationFactor);

		return grid;
	}
}

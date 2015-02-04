package edu.stanford.rsl.conrad.geometry.splines;

import java.util.ArrayList;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class MotionDefectTimeVariantSurfaceBSpline extends
TimeVariantSurfaceBSpline {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1439361309222709769L;

	private PointND min, max;
	private double samplingStep = 0.1;
	private double phaseShift = 0.0;
	private ArrayList<PointND> defectPoints = new ArrayList<PointND>();
	private ArrayList<PointND> interpolationPoints = new ArrayList<PointND>();

	public MotionDefectTimeVariantSurfaceBSpline(
			ArrayList<SurfaceBSpline> splines, PointND min, PointND max) {
		super(splines);
		this.min = new PointND(0,0,0);
		this.max = new PointND(0,0,0);
		// Make sure that min is always smaller than max.
		for (int i =0 ; i < 3; i++){
			this.min.set(i, Math.min(min.get(i), max.get(i)));
			this.max.set(i, Math.max(min.get(i), max.get(i)));
		}
		SurfaceBSpline first = (SurfaceBSpline)timeVariantShapes.get(0);
		for (double v = 0; v < first.vSplines.length; v++){
			ArrayList<PointND> controlPointsU = first.vSplines[(int)v].getControlPoints();
			for (double u = 0; u < controlPointsU.size(); u++){
				boolean inside = true;
				for (int i =0 ; i < 3; i++){
					double coord = controlPointsU.get((int)u).get(i);
					if (!(coord >min.get(i) && coord < max.get(i))){
						inside = false;
						break;
					}
				}
				if (inside) {
					defectPoints.add(new PointND(
							v / first.vSplines.length,
							u / controlPointsU.size()));
					interpolationPoints.add(new PointND(
							v / first.vSplines.length,
							u / controlPointsU.size(), 
							1));
				} else {
					interpolationPoints.add(new PointND(
							v / first.vSplines.length,
							u / controlPointsU.size(), 
							0));
				}
			}
		}
		double sampleBasis = 0.1;
		String key = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_HEART_MOTION_DEFECT_FLEXIBILITY);
		if (key!= null) {
			sampleBasis = Double.parseDouble(key);
		}
		setSamplingStep(sampleBasis);
		key = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_HEART_MOTION_DEFECT_PHASE_SHIFT);
		if (key!= null) {
			phaseShift = Double.parseDouble(key);
		}
	}
	
	public MotionDefectTimeVariantSurfaceBSpline(MotionDefectTimeVariantSurfaceBSpline mds) {
		super(mds);
		min = (mds.min != null) ? mds.min.clone() : null;
		max = (mds.max != null) ? mds.max.clone() : null;
		samplingStep = mds.samplingStep;
		phaseShift = mds.phaseShift;
		
		if (mds.defectPoints != null){
			Iterator<PointND> it = mds.defectPoints.iterator();
			defectPoints = new ArrayList<PointND>();
			while (it.hasNext()) {
				PointND p = it.next();
				defectPoints.add((p!=null) ? p.clone() : null);
			}
		}
		else{
			defectPoints = null;
		}
		
		if (mds.interpolationPoints != null){
			Iterator<PointND> it = mds.interpolationPoints.iterator();
			interpolationPoints = new ArrayList<PointND>();
			while (it.hasNext()) {
				PointND p = it.next();
				interpolationPoints.add((p!=null) ? p.clone() : null);
			}
		}
		else{
			interpolationPoints = null;
		}
	}

	
	private boolean insideDefect(PointND pointUV){
		for (int i=0; i < defectPoints.size(); i++){
			double distance = pointUV.euclideanDistance(defectPoints.get(i));
			if (distance < samplingStep) return true;
		}
		return false;
	}

	private double interpolateWeight(PointND pointUV){
		double weightH = 0;
		double interpolationWeight = 0;
		for (int i= 0; i < interpolationPoints.size(); i++){
			PointND localPoint = interpolationPoints.get(i);
			double distance = Math.sqrt(
					Math.pow(localPoint.get(0) - pointUV.get(0), 2) + 
					Math.pow(localPoint.get(1) - pointUV.get(1), 2));
			double localWeight = Math.exp(-0.5 * Math.pow(distance/samplingStep,2));
			interpolationWeight += interpolationPoints.get(i).get(2) * localWeight; 
			weightH += localWeight;
		}
		return interpolationWeight/weightH;
	}

	public PointND evaluate(double u, double v, double t){
		double [] p = new double [timeVariantShapes.get(0).getControlPoints().get(0).getDimension()];

		double internal = ((t * timeSpline.getKnots().length) -3.0);
		int numPts = timeSpline.getControlPoints().size();
		double step = internal- Math.floor(internal);

		PointND result = null;

		if (insideDefect(new PointND(u,v))){
			PointND pointNotMoving = ((SurfaceBSpline)timeVariantShapes.get(0)).evaluate(u, v);

			if (phaseShift != 0){
				t-=phaseShift;
				t = t%1.0;
				internal = ((t * timeSpline.getKnots().length) -3.0);
				double [] weights = UniformCubicBSpline.getWeights(step);
				double [] p2 = new double [timeVariantShapes.get(0).getControlPoints().get(0).getDimension()];
				for (int i=0;i<4;i++){
					double [] loc = (internal+i < 0)? 
							((SurfaceBSpline)timeVariantShapes.get(0)).evaluate(u, v).getCoordinates()
							: (internal+i>=numPts)? 
									((SurfaceBSpline)timeVariantShapes.get(numPts-1)).evaluate(u, v).getCoordinates() 
									: ((SurfaceBSpline)timeVariantShapes.get((int) (internal+i))).evaluate(u, v).getCoordinates();
									for (int j = 0; j < loc.length; j++){
										p2[j] += (loc[j] * weights[i]);
									}
				}
				pointNotMoving = new PointND(p2);
			}

			// Unbewegte Punkt p_u

			double weight = interpolateWeight(new PointND(u,v));
			double [] weights = UniformCubicBSpline.getWeights(step);
			for (int i=0;i<4;i++){
				double [] loc = (internal+i < 0)? 
						((SurfaceBSpline)timeVariantShapes.get(0)).evaluate(u, v).getCoordinates()
						: (internal+i>=numPts)? 
								((SurfaceBSpline)timeVariantShapes.get(numPts-1)).evaluate(u, v).getCoordinates() 
								: ((SurfaceBSpline)timeVariantShapes.get((int) (internal+i))).evaluate(u, v).getCoordinates();
								for (int j = 0; j < loc.length; j++){
									p[j] += (loc[j] * weights[i]);
								}
			}
			PointND pointMoving = new PointND(p);
			pointMoving.getAbstractVector().multiplyBy(1-weight);
			pointNotMoving .getAbstractVector().multiplyBy(weight);
			pointNotMoving.getAbstractVector().add(pointMoving.getAbstractVector());
			result = pointNotMoving;
		} else {
			// Bewegte Punkt p_b
			double [] weights = UniformCubicBSpline.getWeights(step);
			for (int i=0;i<4;i++){
				double [] loc = (internal+i < 0)? 
						((SurfaceBSpline)timeVariantShapes.get(0)).evaluate(u, v).getCoordinates()
						: (internal+i>=numPts)? 
								((SurfaceBSpline)timeVariantShapes.get(numPts-1)).evaluate(u, v).getCoordinates() 
								: ((SurfaceBSpline)timeVariantShapes.get((int) (internal+i))).evaluate(u, v).getCoordinates();
								for (int j = 0; j < loc.length; j++){
									p[j] += (loc[j] * weights[i]);
								}
			}
			PointND pointMoving = new PointND(p);
			result = pointMoving;
		}

		return result;
	}

	public void setSamplingStep(double samplingStep) {
		this.samplingStep = samplingStep;
	}

	public double getSamplingStep() {
		return samplingStep;
	}

	@Override
	public AbstractShape clone() {
		return new MotionDefectTimeVariantSurfaceBSpline(this);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
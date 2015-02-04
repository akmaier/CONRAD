package edu.stanford.rsl.conrad.geometry.splines;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;


/**
 * Implements a time-variant surface. It is composed of a number of Spline surfaces which are generated from the motion field in the constructor.
 * In the evaluation of the surface the nearest neighbor of the respective point in time is chosen and a point on its surface is returned. 
 * 
 * @author akmaier
 *
 */
public class NearestNeighborTimeVariantSurfaceBSpline extends
		TimeVariantSurfaceBSpline {
	
	public NearestNeighborTimeVariantSurfaceBSpline(NearestNeighborTimeVariantSurfaceBSpline nnsp){
		super(nnsp);
	}

	public NearestNeighborTimeVariantSurfaceBSpline(
			SurfaceBSpline timeInvariant, MotionField motion, int timePoints,
			boolean addInitial) {
		super(timeInvariant, motion, timePoints, addInitial);
		// TODO Auto-generated constructor stub
	}

	private static final long serialVersionUID = 7390783547660669266L;
	
	public PointND evaluate(double u, double v, double t){
		// time is given by the projector or renderer in the range of [0, (#Projections-1)/#Projections]
		// to retrieve #Projections time variant shapes we internal = (int) Math.round((t * (tPoints)));
		// Because the NearestNeighborTimeVariantSurfaceBSpline is either initialized by 
		// 		(1) timePoints = #Projections+1 and addInitial = false
		//OR	(2) timePoints = #Projections 	and addInitial = true
		// Because of the superclass constructor, both cases yield a tPoints variable of #Projections
		int internal = (int) Math.round((t * (tPoints)));
		return ((SurfaceBSpline)timeVariantShapes.get(internal)).evaluate(u, v);
	}
	
	@Override
	public AbstractShape clone() {
		return new NearestNeighborTimeVariantSurfaceBSpline(this);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
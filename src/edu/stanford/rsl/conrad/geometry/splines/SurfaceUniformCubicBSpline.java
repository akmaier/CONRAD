package edu.stanford.rsl.conrad.geometry.splines;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class SurfaceUniformCubicBSpline extends SurfaceBSpline {

	public SurfaceUniformCubicBSpline(ArrayList<PointND> list, double[] uKnots,
			double[] vKnots) {
		super(list, uKnots, vKnots);
	}

	public SurfaceUniformCubicBSpline(String last, ArrayList<PointND> list,
			double[] uKnots, double[] vKnots) {
		super(last, list, uKnots, vKnots);
	}

	public SurfaceUniformCubicBSpline(String name,
			ArrayList<PointND> controlPoints, SimpleVector uKnots,
			SimpleVector vKnots) {
		super(name, controlPoints,uKnots, vKnots);
	}

	public SurfaceUniformCubicBSpline(ArrayList<PointND> initialPoints,
			SimpleVector uKnots, SimpleVector vKnots) {
		super(initialPoints, uKnots, vKnots);
	}

	public SurfaceUniformCubicBSpline(SurfaceBSpline spline) {
		super(spline);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -7498785267911679836L;

	protected synchronized void init(){
		dimension = points.get(0).getDimension();
		int degreeU = 0;
		while(uKnots.getElement(degreeU) == 0) degreeU++;
		degreeU --;
		int degreeV = 0;
		while(uKnots.getElement(degreeV) == 0) degreeV++;
		degreeV --;
		numberOfUPoints = uKnots.getLen() - ((degreeU+1));
		numberOfVPoints = vKnots.getLen() - ((degreeV+1));
		assert(numberOfUPoints * numberOfVPoints == points.size());
		ArrayList<PointND> temp = new ArrayList<PointND>();
		for(int i = 0; i < numberOfUPoints; i++){
			temp.add(points.get(0));
		}
		uSpline = new UniformCubicBSpline(temp, uKnots); //TODO: decrease duplicate code (Generics<T>)
		vSplines = new BSpline[numberOfUPoints];
		for(int i = 0; i < numberOfUPoints; i++){
			temp = new ArrayList<PointND>();
			for(int j = 0; j < numberOfVPoints; j++){
				temp.add(points.get((i*numberOfVPoints)+j));
			}
			vSplines[i] = new UniformCubicBSpline(temp, vKnots);
		}
		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		for (PointND p: points) {
			max.updateIfHigher(p);
			min.updateIfLower(p);
		}
		generateBoundingPlanes();
	}
	
	@Override
	public PointND evaluate(double u, double v){
		double [] p = new double [getControlPoints().get(0).getDimension()];

		double internal = ((u * uSpline.getKnots().length) -3.0);
		int numPts = uSpline.getControlPoints().size();
		double step = internal- Math.floor(internal);

		//System.out.println(u + " " + internal);
		double [] weights = UniformCubicBSpline.getWeights(step);
		for (int i=0;i<4;i++){
			double [] loc = (internal+i < 0)? vSplines[0].evaluate(v).getCoordinates(): (internal+i>=numPts)? vSplines[numPts-1].evaluate(v).getCoordinates() : vSplines[(int) (internal+i)].evaluate(v).getCoordinates();
			for (int j = 0; j < loc.length; j++)
				p[j] += (loc[j] * weights[i]);
				//p[j] =weights[i];
				//p[j] = loc[j];
		}
		return new PointND(p);
	}
	
	@Override
	public AbstractShape clone() {
		return new SurfaceUniformCubicBSpline(this);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
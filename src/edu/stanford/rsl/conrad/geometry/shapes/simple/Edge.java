package edu.stanford.rsl.conrad.geometry.shapes.simple;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class Edge extends StraightLine {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3399981161692525136L;
	protected double end;
	protected PointND endPoint;
	private double length;

	public Edge(PointND point, PointND point2) {
		super(point, point2);
		end = 1;
		endPoint = point2;
		length = point.euclideanDistance(point2);
		this.min = new PointND(point);
		this.max = new PointND(point);
		for (int i=0; i < point.getDimension();i++){
			min.set(i, Math.min(point.get(i), point2.get(i)));
			max.set(i, Math.max(point.get(i), point2.get(i)));
		}
	}
	
	public Edge(Edge e){
		super(e);
		end = e.end;
		length = e.length;
		endPoint = (e.endPoint != null) ? e.endPoint.clone() : null;
	}

	public void setEnds(PointND point, PointND point2){
		super.init(point, point2);
		end = 1;
		endPoint = point2;
		length = point.euclideanDistance(point2);

		for (int i=0; i < point.getDimension();i++){
			min.set(i, Math.min(point.get(i), point2.get(i)));
			max.set(i, Math.max(point.get(i), point2.get(i)));
		}
	}

	public double getLastInternalIndex(){
		return end;
	}

	public PointND getEnd(){
		return endPoint;
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	@Override
	public PointND intersect(StraightLine line) {
		try {
			SimpleVector lambda = computeIntersectionCoefficients(line);
			if (!((lambda.getElement(0) < -CONRAD.FLOAT_EPSILON) || (lambda.getElement(0) >= end + CONRAD.FLOAT_EPSILON))){
				PointND p1 = evaluate(lambda.getElement(0));
				PointND p2 = line.evaluate(lambda.getElement(1));
				if (p1 != null) {
					if (p1.euclideanDistance(p2) < CONRAD.SMALL_VALUE){
						return p1;
					} 
				}
			} 
		} catch (IllegalArgumentException e){
			System.err.println("edu.stanford.rsl.conrad.geometry.shapes.simple.Edge Line 69: Intersection with degenerated triangle produced no intersection" + e.getMessage());
		}
		return null;
	}

	@Override
	public PointND evaluate(double u){
		if (u <= end){
			return super.evaluate(u);
		} else {
			return null;
		}
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof Edge) {
			Edge e = (Edge) o;
			return ((getEnd().equals(e.getEnd()) && getPoint().equals(e.getPoint())) ||
					(getPoint().equals(e.getEnd()) && getEnd().equals(e.getPoint())));
			//return (point == e.endPoint && endPoint == e.point) ||
			//(endPoint == e.endPoint && point == e.point);
		} else {
			return false;
		}
	}

	@Override
	public String toString(){
		return getPoint() + " " + getEnd();
	}

	public PointND[] getRasterPoints(int number){
		PointND [] points = new PointND [number];
		double step = end / (number - 1.0);
		//System.out.println(step + " " + number);
		points[0] = new PointND(point);
		points[number-1] = new PointND(endPoint);
		//SimpleVector increment = direction.multipliedBy(step);
		for (int i = number -2; i > 0; i--){
			//System.out.println(i + " " + step + " "+  i*step + "points[i] = " + evaluate(i*step));
			points[i] = evaluate(i*step);

		}
		return points;
	}

	@Override
	public void applyTransform(Transform t){
		super.applyTransform(t);
		endPoint = t.transform(endPoint);

		this.min = t.transform(this.min);
		this.max = t.transform(this.max);
	}

	/**
	 * Returns length of edge in mm
	 * @return the length
	 */
	public double getLength(){
		return length;
	}

	@Override
	public AbstractShape clone() {
		return new Edge(this);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
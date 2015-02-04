package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to describe a triangle in 3D.
 * 
 * @author akmaier
 *
 */
public class Triangle extends Plane3D {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3360323076084779150L;
	protected double bUcoord, bVcoord;
	protected double cUcoord, cVcoord;

	/**
	 * Creates a new Triangle from the Points a, b, and c
	 * @param a
	 * @param b
	 * @param c
	 */
	public Triangle(PointND a, PointND b, PointND c){
		super(a, SimpleOperators.subtract(b.getAbstractVector(), a.getAbstractVector()), SimpleOperators.subtract(c.getAbstractVector(), a.getAbstractVector()));
		bUcoord = 1;
		bVcoord = 0;
		cUcoord = 0;
		cVcoord = 1;
		updateBounds(a, b, c);
	}

	public Triangle(Triangle shape){
		super(shape);
		bUcoord = shape.bUcoord;
		bVcoord = shape.bVcoord;
		cUcoord = shape.cUcoord;
		cVcoord = shape.cVcoord;
	}

	@Override
	public ArrayList<PointND> getHitsOnBoundingBox(AbstractCurve other){
		return intersect(other);
	}

	/**
	 * Returns point a
	 * @return point a
	 */
	public PointND getA() {
		return new PointND(pointP);
	}

	/**
	 * Returns point b
	 * @return point b
	 */
	public PointND getB() {
		return evaluate(bUcoord, bVcoord);
	}

	/**
	 * Returns point c
	 * @return point c
	 */
	public PointND getC() {
		return evaluate(cUcoord, cVcoord);
	}

	@Override
	public PointND intersect(StraightLine other) {
		PointND revan = super.intersect(other);
		//System.out.println(revan);
		if (isInTriangle(revan)){
			return revan;
		} else {
			return null;
		}
	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		if (other instanceof StraightLine) {
			try {
				ArrayList<PointND> list = new ArrayList<PointND>();
				PointND p = intersect((StraightLine) other);
				if (p!= null) list.add(p);
				return list;
			} catch (RuntimeException e){
				if (e.getLocalizedMessage().equals("Line is parallel to plane")){
					ArrayList<PointND> list = new ArrayList<PointND>();
					Edge one = new Edge(getA(), getB());
					PointND p = one.intersect((StraightLine) other);
					if (p != null) list.add(p);
					one = new Edge(getB(), getC());
					p = one.intersect((StraightLine) other);
					if (p != null) list.add(p);
					one = new Edge(getA(), getC());
					p = one.intersect((StraightLine) other);
					if (p != null) list.add(p);
					return list;
				} else {
					throw(e);
				}
			}
		} else {
			throw new RuntimeException("Not implemented yet!");
		}
	}

	/**
	 * Computes whether the given point is inside of the triangle. Implementation is based on barycentric coordinates.
	 *  
	 * @param p the point
	 * @return true if it is inside of the triangle.
	 */
	public boolean isInTriangle(PointND p){
		// vectors  
		SimpleVector v0 = dirV;
		SimpleVector v1 = dirU;
		SimpleVector v2 = SimpleOperators.subtract(p.getAbstractVector(), pointP.getAbstractVector());
		// Compute dot products

		double dot00  = SimpleOperators.multiplyInnerProd(v0, v0);
		double dot01  = SimpleOperators.multiplyInnerProd(v0, v1);
		double dot02  = SimpleOperators.multiplyInnerProd(v0, v2);
		double dot11  = SimpleOperators.multiplyInnerProd(v1, v1);
		double dot12  = SimpleOperators.multiplyInnerProd(v1, v2);

		// Compute barycentric coordinates
		double invDenom = 1.0 / ((dot00 * dot11) - (dot01 * dot01));
		double u = ((dot11 * dot02) - (dot01 * dot12)) * invDenom;
		double v = ((dot00 * dot12) - (dot01 * dot02)) * invDenom;
		return (u >= 0) && (v >= 0) && (u + v <= 1);
	}

	public PointND[] getRasterPoints(int number){
		Edge one = new Edge (getA(), getB());
		Edge two = new Edge (getB(), getC());
		Edge three = new Edge (getC(), getA());
		//System.out.println(getA() + " " + getB() + " " + one.getLastInternalIndex());
		PointND [] points1 = one.getRasterPoints(number /3);
		//System.out.println("End");
		//System.exit(-1);
		PointND [] points2 = two.getRasterPoints(number /3);
		PointND [] points3 = three.getRasterPoints(number /3);
		PointND [] points = new PointND[points1.length + points2.length + points3.length];
		System.arraycopy(points1, 0, points, 0, points1.length);
		System.arraycopy(points2, 0, points, points1.length, points2.length);
		System.arraycopy(points3, 0, points, points1.length + points2.length, points3.length);
		return points;
	}

	@Override
	protected void generateBoundingPlanes(){
		// Nothing to do here.
	}

	@Override
	public boolean isBounded(){
		return true;
	}

	@Override
	public void applyTransform(Transform t) {
		SimpleVector buff = t.transform(normalN);
		normalN = buff.dividedBy(buff.normL2());
		pointP = t.transform(pointP);
		offsetD = SimpleOperators.multiplyInnerProd(this.normalN, this.pointP.getAbstractVector());
		dirU = t.transform(dirU);
		dirV = t.transform(dirV);
		updateBounds();
	}

	@Override
	public String toString(){
		return "Triangle " + this.getName() + ": " + getA() + " " + getB() + " " + getC();
	}

	private void updateBounds(){
		updateBounds(getA(),getB(),getC());
	}

	private void updateBounds(PointND a, PointND b, PointND c){
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		min.updateIfLower(a);
		min.updateIfLower(b);
		min.updateIfLower(c);
		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		max.updateIfHigher(a);
		max.updateIfHigher(b);
		max.updateIfHigher(c);
	}

	@Override
	public AbstractShape clone() {
		return new Triangle(this);
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
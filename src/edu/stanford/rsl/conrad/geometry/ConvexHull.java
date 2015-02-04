package edu.stanford.rsl.conrad.geometry;

import java.util.ArrayList;
import java.util.Stack;

import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Point3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/** super class for hull constructing algorithms
 * 
 * Based on the gift wrapping code from Tim Lambert. Demo applets are presented <a href="http://www.cse.unsw.edu.au/~lambert/java/3d/hull.html">here</a>.
 * 
 */

public class ConvexHull {

	protected PointND[] pts;

	protected PointND [] hullPoints;

	protected Triangle [] faces;

	int[] extraColors(){
		return new int[0];
	}
	public ConvexHull(PointND[] pts) {
		this.pts = pts;
	}

	int index(PointND p) {
		for(int i=0; i<pts.length; i++){
			if (p==pts[i]) {
				return i;
			}
		}
		return -1;
	}

	protected PointND search(Edge line) {
		int i;
		PointND p1 = line.getPoint();
		PointND p2 = line.getEnd();
		for(i = 0; pts[i].equals(p1) || pts[i].equals(p2); i++) {
			/* nothing */
		}
		//System.out.println("Found at " + i);
		PointND cand = pts[i];
		SimpleVector edgeDirection = SimpleOperators.subtract(p2.getAbstractVector(), p1.getAbstractVector());
		SimpleVector otherDirection = SimpleOperators.subtract(cand.getAbstractVector(), p1.getAbstractVector());
		HalfSpaceBoundingCondition candh = new HalfSpaceBoundingCondition(new Plane3D(p1, edgeDirection, otherDirection));
		for(i=i+1; i < pts.length; i++) {
			if ((!pts[i].equals(p1)) && (!pts[i].equals(p2)) && candh.isSatisfiedBy(pts[i])) {
				cand = pts[i];
				otherDirection = SimpleOperators.subtract(cand.getAbstractVector(), p1.getAbstractVector());
				candh = new HalfSpaceBoundingCondition(new Plane3D(p1,edgeDirection,otherDirection));
			}
		}
		return cand;
	}

	protected PointND search2d(PointND p) {
		int i;
		Point3D k = new Point3D(0,0,1);
		i = pts[0] == p?1:0;
		PointND cand = pts[i];
		SimpleVector edgeDirection = SimpleOperators.subtract(cand.getAbstractVector(), p.getAbstractVector());
		SimpleVector otherDirection = k.getAbstractVector();
		HalfSpaceBoundingCondition candh = new HalfSpaceBoundingCondition(new Plane3D(cand, edgeDirection, otherDirection));
		for(i=i+1; i < pts.length; i++) {
			if (pts[i] != p && candh.isSatisfiedBy(pts[i])) {
				cand = pts[i];
				edgeDirection = SimpleOperators.subtract(cand.getAbstractVector(), p.getAbstractVector());
				candh = new HalfSpaceBoundingCondition(new Plane3D(cand, edgeDirection, otherDirection));
			}
		}
		return cand;
	}

	/* bottom point */
	protected PointND bottom(){
		PointND bot = pts[0];
		for (int i = 1; i < pts.length; i++) {
			if (pts[i].get(1) < bot.get(1)) {
				bot = pts[i];
			}
		}
		return bot;
	}

	public void build () {
		/* First find a hull edge -- just connect bottommost to second from bottom */
		PointND bot, bot2; /* bottom point and adjacent point*/
		bot = bottom();
		bot2 = search2d(bot);
		ArrayList<PointND> hull = new ArrayList<PointND>();
		hull.add(bot);
		hull.add(bot2);
		/* intialize the edge stack */
		Stack<Edge> es = new Stack<Edge>();
		Stack<Edge> done = new Stack<Edge>();
		es.push(new Edge(bot,bot2));	
		es.push(new Edge(bot2,bot));
		ArrayList<AbstractShape> faces = new ArrayList<AbstractShape>();
		Edge e = null;
		System.out.println(pts.length);
		//System.exit(-1);
		
		/* now the main loop -- keep finding faces till there are no more to be found */
		while (! es.isEmpty() ) {
			e = es.pop();
			PointND cand = search(e);
			hull.add(cand);

			Triangle t = new Triangle(e.getPoint(),cand,e.getEnd());

			faces.add(t);

			Edge edge1 = new Edge(e.getPoint(),cand);
			Edge edge2 = new Edge(cand,e.getEnd());

			if (es.contains(edge1)) {
				es.remove(edge1);
				done.add(edge1);
			} else {
				if (! done.contains(edge1)) {
					es.push(edge1);
				}
			}

			if (es.contains(edge2)) {
				es.remove(edge2);
				done.add(edge2);
			} else {
				if (!done.contains(edge2)) {
					es.push(edge2);
				}
			}
			System.out.println("current size "  + es.size() + " "  + faces.size());
		}
		hullPoints = new PointND[hull.size()];
		hullPoints = hull.toArray(hullPoints);
		this.faces = new Triangle [faces.size()];
		this.faces = faces.toArray(this.faces);
	}


	public void build2D() {
		/* First find a hull vertex -- just bottommost*/
		PointND p; /* current hull vertex */
		PointND bot = bottom(); /* bottom point */

		ArrayList<PointND> hull = new ArrayList<PointND>();
		hull.add(bot);

		/* now the main loop -- keep finding edges till we get back */

		p = bot;
		do {
			PointND cand = search2d(p);
			hull.add(cand);
			p = cand;
		} while (p!=bot);
		hullPoints = new PointND[hull.size()];
		hullPoints = hull.toArray(hullPoints);
		PointND center = General.getGeometricCenter(hull);
		PointND offCenter = new PointND(center);
		offCenter.set(2, offCenter.get(2)+10);
		faces = new Triangle [hullPoints.length-1];
		for (int i = 1; i < hullPoints.length; i++){
			faces[i-1] = new Triangle(offCenter, hullPoints[i], hullPoints[i-1]);
			//System.out.println("Center distance 1 " + faces[i-1].computeDistance(center) + " " + faces[i-1].normalN);
			//if (faces[i-1].computeDistance(center) < 0) {
			//	faces[i-1].normalN.negate();
			//	faces[i-1].offsetD *= -1;
			//}
			//System.out.println("Center distance 2 " + faces[i-1].computeDistance(center) + " " + faces[i-1].normalN + "\n");

		}
	}

	/**
	 * returns the hull as an array of Points
	 * @return the array of Points
	 */
	public PointND [] getHullPoints(){
		return hullPoints;
	}

	/**
	 * Returns the hull as an array of triangles
	 * 
	 * @return the array of faces
	 */
	public Triangle [] getFaces(){
		return faces;
	}

	/**
	 * Tests whether the point is inside the convex hull.
	 * @param point
	 * @return true if the point is inside
	 */
	public boolean isInside(PointND point){
		boolean revan = true;
		for (int i=1; i< faces.length; i++){
			Triangle t =  faces[i];
			HalfSpaceBoundingCondition bound = new HalfSpaceBoundingCondition(t);
			if (!bound.isSatisfiedBy(point)) {
				revan = false;
				System.out.println(point +  " " + t.computeDistance(point) + " " + i +" "+ t.getNormal() + " " + t.getPoint());
				break;
			}
		}
		return revan;
	}

	/**
	 * Intersects lines between subsequent hull points. Only applicable in 2D.
	 * @param line the line to intersect with the hull
	 * @return the array of intersection points.
	 */
	public PointND [] intersect2D(StraightLine line){
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i = 1; i < hullPoints.length; i++){
			Edge test = new Edge(hullPoints[i-1], hullPoints[i]);
			PointND p = test.intersect(line);
			if (p != null){
				list.add(p);
			}
		}
		PointND [] points =  new PointND [list.size()];
		points = list.toArray(points);
		return points;
	}

	public PointND [] intersect3D(StraightLine line){
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i = 0; i < faces.length; i++){
			PointND p = faces[i].intersect(line);
			//System.out.println("Intersection " + i + " " + p);
			if (p != null){
				list.add(p);
			}
		}
		PointND [] points =  new PointND [list.size()];
		points = list.toArray(points);
		return points;
	}

	public PointND[] getRasterPoints(int number){
		ArrayList<PointND[]> lists = new ArrayList<PointND[]>();
		int sum = 0;
		for (Triangle t : faces){
			PointND [] pts = t.getRasterPoints(number/ faces.length);
			sum += pts.length;
			lists.add(pts);
		}
		PointND [] points = new PointND[sum];
		int increment = 0;
		for (PointND [] pts : lists){
			System.arraycopy(pts, 0, points, increment, pts.length);
			increment += pts.length;
		}
		return points;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
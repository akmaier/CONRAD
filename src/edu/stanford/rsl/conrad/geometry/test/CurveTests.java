package edu.stanford.rsl.conrad.geometry.test;

import junit.framework.Assert;

import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class CurveTests {

	
	@Test
	public void intersectTest(){
		PointND one = new PointND(0, 0, 0);
		PointND two = new PointND(1, 1, 1);
		PointND three = new PointND(1, 0, 0);
		StraightLine line1 = new StraightLine(one, two);
		StraightLine line2 = new StraightLine(two, three);
		StraightLine line3 = new StraightLine(one, new SimpleVector(-1,-1,-1));
		Triangle t = new Triangle(one, new PointND(1,1, 0), three);
		Assert.assertEquals(t.intersect(line1).euclideanDistance(t.intersect(line3)), 0.0);
		Assert.assertEquals(line1.intersect(line2).euclideanDistance(two), 0.0);
		Assert.assertEquals(line3.intersect(line2).euclideanDistance(two), 0.0);
	}
	
	@Test
	public void evaluateTest(){
		PointND one = new PointND(0, 0, 0);
		PointND two = new PointND(1, 1, 1);
		StraightLine line1 = new StraightLine(one, two);
		Assert.assertEquals(line1.evaluate(1.0).euclideanDistance(two), 0.0);
	}
	
	@Test
	public void edgeConstructionTest(){
		PointND one = new PointND(0, 0, -1);
		PointND two = new PointND(2, 2, 0);
		Edge line1 = new Edge(one, two);
		Assert.assertEquals(two, line1.getEnd());
		//System.out.println(line1.evaluate(line1.getLastInternalIndex()));
		Assert.assertEquals(two, line1.evaluate(line1.getLastInternalIndex()));
		Assert.assertEquals(one, line1.getPoint());
		
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
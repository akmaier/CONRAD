package edu.stanford.rsl.conrad.geometry.test;

import junit.framework.Assert;

import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;

public class SurfaceTests {

	@Test
	public void intersectPlaneTest(){
		PointND one = new PointND(0, 0, 0);
		PointND two = new PointND(1, 1, 1);
		PointND three = new PointND(1, 0, 0);
		PointND result = new PointND(0, 1, 1);
		StraightLine line = new StraightLine(two, three.getAbstractVector());
		Plane3D plane = new Plane3D(one, three.getAbstractVector());
		Assert.assertEquals(plane.intersect(line).euclideanDistance(result), 0.0);
	}
	
	@Test
	public void triangleIntersectionTest(){
		PointND one = new PointND(0, 0, 0);
		PointND two = new PointND(1, 0, 0);
		PointND three = new PointND(1, 1, 0);
		PointND inside = new PointND(0.9, 0.1, 0);
		PointND outside = new PointND(0.1, 0.9, 0);
		PointND result = new PointND(1, 1, 3);
		Edge through = new Edge(inside, result);
		Edge pass = new Edge(outside, result);
		Triangle triangle = new Triangle(one, two, three);
		Assert.assertEquals(triangle.getA(), one);
		Assert.assertEquals(triangle.getB(), two);
		Assert.assertEquals(triangle.getC(), three);
		PointND nohit = triangle.intersect(pass);
		Assert.assertEquals(null, nohit);
		PointND hit = triangle.intersect(through);
		Assert.assertEquals(inside, hit);
	}
	
	@Test
	public void translateTest(){
		PointND one = new PointND(0, 0, 0);
		PointND two = new PointND(1, 0, 0);
		PointND three = new PointND(1, 1, 0);
		Transform transform = new Translation(1,1,1);
		PointND oneprime = transform.transform(one);
		PointND twoprime = transform.transform(two);
		PointND threeprime = transform.transform(three);
		Triangle triangle = new Triangle(one,two,three);
		triangle.applyTransform(transform);
		Assert.assertEquals(oneprime, triangle.getA());
		Assert.assertEquals(twoprime, triangle.getB());
		Assert.assertEquals(threeprime, triangle.getC());
	}
	
	@Test
	public void rotateTest(){
		PointND one = new PointND(0, 0, 0);
		PointND two = new PointND(1, 0, 0);
		PointND three = new PointND(1, 1, 0);
		Transform transform = new ScaleRotate(Rotations.createRotationMatrixAboutAxis(new Axis(1,1,1), 0.5));
		PointND oneprime = transform.transform(one);
		PointND twoprime = transform.transform(two);
		PointND threeprime = transform.transform(three);
		Triangle triangle = new Triangle(one,two,three);
		triangle.applyTransform(transform);
		Assert.assertEquals(oneprime, triangle.getA());
		Assert.assertEquals(twoprime, triangle.getB());
		Assert.assertEquals(threeprime, triangle.getC());
	}
	
	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import org.junit.Assert;
import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class GeometryTests {

	@Test
	public void testBoxIntersectionSimple(){
		Box box = new Box(1,1,1);
		StraightLine line = new StraightLine(new PointND (-10, 0.5 ,0.5), new SimpleVector(1, 0, 0));
		ArrayList<PointND> intersections = box.intersect(line);
		PointND in = new PointND(0.0, 0.5, 0.5);
		PointND out = new PointND(1.0, 0.5, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, 0.75 ,0.5), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0.0, 0.75, 0.5);
		out = new PointND(1.0, 0.75, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, 0.25 ,0.5), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0.0, 0.25, 0.5);
		out = new PointND(1.0, 0.25, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
	@Test
	public void testBoxIntersection90DegreeRotation(){
		Box box = new Box(1,1,1);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/2));
		box.applyTransform(r);
		StraightLine line = new StraightLine(new PointND (-10, 0.5 ,0.5), new SimpleVector(1, 0, 0));
		ArrayList<PointND> intersections = box.intersect(line);
		PointND in = new PointND(0.0, 0.5, 0.5);
		PointND out = new PointND(-1.0, 0.5, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, 0.75 ,0.5), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0.0, 0.75, 0.5);
		out = new PointND(-1.0, 0.75, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, 0.25 ,0.5), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0.0, 0.25, 0.5);
		out = new PointND(-1.0, 0.25, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
	@Test
	public void testBoxIntersection90DegreeRotationAndShift(){
		Box box = new Box(1,1,1);
		Translation t = new Translation(new SimpleVector(-0.5,-0.5,-0.5));
		box.applyTransform(t);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/2));
		box.applyTransform(r);
		StraightLine line = new StraightLine(new PointND (-10, 0.0 ,0.0), new SimpleVector(1, 0, 0));
		ArrayList<PointND> intersections = box.intersect(line);
		PointND in = new PointND(0.5, 0.0, 0.0);
		PointND out = new PointND(-0.5, 0.0, 0.0);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, -0.25 ,0.5), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0.5, -0.25, 0.5);
		out = new PointND(-.50, -0.25, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, 0.25 ,0.5), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0.5, 0.25, 0.5);
		out = new PointND(-.50, 0.25, 0.5);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
	@Test
	public void testBoxIntersectionAndShift(){
		Box box = new Box(1,1,1);
		Translation t = new Translation(new SimpleVector(-0.5,-0.5,-0.5));
		box.applyTransform(t);
		for (int i = 0; i < 200; i++){
			double y = i/10.0;
			StraightLine line = new StraightLine(new PointND (-10, y ,0.0), new SimpleVector(1, 0, 0));
			ArrayList<PointND> intersections = box.intersect(line);
			PointND in = new PointND(0.5, y, 0.0);
			PointND out = new PointND(-0.5, y, 0.0);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			line = new StraightLine(new PointND (-10, -0.25 +y,0.5), new SimpleVector(1, 0, 0));
			intersections = box.intersect(line);
			in = new PointND(0.5, -0.25+y, 0.5);
			out = new PointND(-.50, -0.25+y, 0.5);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			line = new StraightLine(new PointND (-10, 0.25+y ,0.5), new SimpleVector(1, 0, 0));
			intersections = box.intersect(line);
			in = new PointND(0.5, 0.25+y, 0.5);
			out = new PointND(-.50, 0.25+y, 0.5);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			t = new Translation(new SimpleVector(0,0.1,0));
			box.applyTransform(t);
		}
	}
	
	@Test
	public void testBoxIntersectionRotationAndShift(){
		Box box = new Box(1,1,1);
		Translation t = new Translation(new SimpleVector(-0.5,-0.5,-0.5));
		box.applyTransform(t);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/2));
		box.applyTransform(r);
		for (int i = 0; i < 200; i++){
			double y = -i/10.0;
			box.applyTransform(r);
			t = new Translation(new SimpleVector(0,y,0));
			box.applyTransform(t);
			StraightLine line = new StraightLine(new PointND (-10, y ,0.0), new SimpleVector(1, 0, 0));
			ArrayList<PointND> intersections = box.intersect(line);
			PointND in = new PointND(0.5, y, 0.0);
			PointND out = new PointND(-0.5, y, 0.0);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			line = new StraightLine(new PointND (-10, -0.25 +y,0.5), new SimpleVector(1, 0, 0));
			intersections = box.intersect(line);
			in = new PointND(0.5, -0.25+y, 0.5);
			out = new PointND(-.50, -0.25+y, 0.5);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			line = new StraightLine(new PointND (-10, 0.25+y ,0.5), new SimpleVector(1, 0, 0));
			intersections = box.intersect(line);
			in = new PointND(0.5, 0.25+y, 0.5);
			out = new PointND(-.50, 0.25+y, 0.5);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			t = new Translation(new SimpleVector(0,-y,0));
			box.applyTransform(t);
		}
	}
	
	@Test
	public void testBoxIntersectionRotationAndShiftXYZ(){
		Box box = new Box(1,1,1);
		Translation t = new Translation(new SimpleVector(-0.5,-0.5,-0.5));
		box.applyTransform(t);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/2));
		box.applyTransform(r);
		for (int i = 0; i < 200; i++){
			double x = (Math.random() - 0.5)*50;
			double y = (Math.random() - 0.5)*50;
			double z = (Math.random() - 0.5)*50;
			box.applyTransform(r);
			t = new Translation(new SimpleVector(x,y,z));
			box.applyTransform(t);
			StraightLine line = new StraightLine(new PointND (-10+x, y ,z), new SimpleVector(1, 0, 0));
			ArrayList<PointND> intersections = box.intersect(line);
			PointND in = new PointND(0.5+x, y, z);
			PointND out = new PointND(-0.5+x, y, z);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			line = new StraightLine(new PointND (-10+x, -0.25 +y,0.5+z), new SimpleVector(1, 0, 0));
			intersections = box.intersect(line);
			in = new PointND(0.5+x, -0.25+y, 0.5+z);
			out = new PointND(-.5+x, -0.25+y, 0.5+z);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			line = new StraightLine(new PointND (-10+x, 0.25+y ,0.5+z), new SimpleVector(1, 0, 0));
			intersections = box.intersect(line);
			in = new PointND(0.5+x, 0.25+y, 0.5+z);
			out = new PointND(-.5+x, 0.25+y, 0.5+z);
			Assert.assertTrue(intersections.contains(in));
			Assert.assertTrue(intersections.contains(out));
			t = new Translation(new SimpleVector(-x,-y,-z));
			box.applyTransform(t);
		}
	}
	
	
	@Test
	public void testBoxIntersection45DegreeRotationAndShift(){
		Box box = new Box(1,1,1);
		Translation t = new Translation(new SimpleVector(-0.5,-0.5,-0.5));
		box.applyTransform(t);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/4));
		box.applyTransform(r);
		StraightLine line = new StraightLine(new PointND (-10, 0.0 ,0.0), new SimpleVector(1, 0, 0));
		ArrayList<PointND> intersections = box.intersect(line);
		PointND in = new PointND(Math.sqrt(2)/2, 0.0, 0.0);
		PointND out = new PointND(-Math.sqrt(2)/2, 0.0, 0.0);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
		line = new StraightLine(new PointND (-10, -Math.sqrt(2)/2 ,0.0), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0, -Math.sqrt(2)/2, 0.0);
		Assert.assertTrue(intersections.contains(in));
		line = new StraightLine(new PointND (-10, Math.sqrt(2)/2 ,0.0), new SimpleVector(1, 0, 0));
		intersections = box.intersect(line);
		in = new PointND(0, Math.sqrt(2)/2, 0.0);
		Assert.assertTrue(intersections.contains(in));
	}
	
	@Test
	public void testCompoundShapeIntersection(){
		AbstractShape linepair = generateLP(0.5);
		StraightLine line = new StraightLine(new PointND (-10, 0.0 ,0.0), new SimpleVector(1, 0, 0));
		ArrayList<PointND> intersections = linepair.intersect(line);
		PointND in = new PointND(-2.25, 0.0, 0.0);
		PointND out = new PointND(2.25, 0.0, 0.0);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
	@Test
	public void testCompoundShapeIntersection90DegreeRotation(){
		AbstractShape linepair = generateLP(0.5);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/2));
		linepair.applyTransform(r);
		StraightLine line = new StraightLine(new PointND (0, -10.0 ,0.0), new SimpleVector(0, 1, 0));
		ArrayList<PointND> intersections = linepair.intersect(line);
		PointND in = new PointND(0.0,2.25, 0.0);
		PointND out = new PointND(0.0, -2.25, 0.0);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
	public CompoundShape generateLP(double gapSize){
		int blocks = 5;
		if (gapSize >= 1) blocks = 4;
		if (gapSize >= 2.5) blocks = 3;
		if (gapSize >= 5) blocks = 2;
		double width = blocks * gapSize + (blocks-1) * gapSize;
		double height = 5;
		double centerX = width / 2;
		double centerY = height / 2;
		CompoundShape lp = new CompoundShape();
		for (int i = 0; i < blocks; i++){
			Box block = new Box(gapSize, height, height);
			block.applyTransform(new Translation((-centerX)+(i*(2*gapSize)), -centerY, -centerY));
			lp.add(block);
		}
		return lp;
	}
	
	@Test
	public void testCompoundShapeIntersection45DegreeRotation(){
		AbstractShape linepair = generateLP(0.5);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/4));
		linepair.applyTransform(r);
		StraightLine line = new StraightLine(new PointND (-10, -10.0 ,0.0), new SimpleVector(1, 1, 0));
		ArrayList<PointND> intersections = linepair.intersect(line);
		PointND in = new PointND(2.25 * (Math.sqrt(2)/2),2.25 * (Math.sqrt(2)/2), 0.0);
		PointND out = new PointND(-2.25 * (Math.sqrt(2)/2), -2.25 * (Math.sqrt(2)/2), 0.0);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
	@Test
	public void testCompoundShapeIntersection45DegreeRotationAndShift(){
		AbstractShape linepair = generateLP(0.5);
		ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/4));
		linepair.applyTransform(r);
		Translation t = new Translation(new SimpleVector(0.0,0.0, 50));
		linepair.applyTransform(t);
		StraightLine line = new StraightLine(new PointND (-10, -10.0 ,50), new SimpleVector(1, 1, 0));
		ArrayList<PointND> intersections = linepair.intersect(line);
		PointND in = new PointND(2.25 * (Math.sqrt(2)/2),2.25 * (Math.sqrt(2)/2), 50.0);
		PointND out = new PointND(-2.25 * (Math.sqrt(2)/2), -2.25 * (Math.sqrt(2)/2), 50.0);
		Assert.assertTrue(intersections.contains(in));
		Assert.assertTrue(intersections.contains(out));
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
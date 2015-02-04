package edu.stanford.rsl.conrad.geometry.test;

import ij.ImageJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

import java.util.ArrayList;

import junit.framework.Assert;

import org.junit.Test;

import edu.stanford.rsl.conrad.geometry.ConvexHull;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Point3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.CircularTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.utils.DoublePrecisionPointUtil;

public class GeometryTests {

	
	@Test
	public void testConvexHull2D(){
		int width = 480;
		int height = 480;
		FloatProcessor test = new FloatProcessor(width, height);
		PointND [] points = new PointND[2000];
		for (int i = 0; i < points.length; i++){
			points[i] = new PointND(new SimpleVector(3));
			points[i].getAbstractVector().randomize(0, height*0.8);
			points[i].getAbstractVector().add(height*0.05);
			points[i].getAbstractVector().setElementValue(2, 0);
			test.putPixelValue((int)(points[i].get(0)), (int)(points[i].get(1)), 10.0);
		}
		ConvexHull hull = new ConvexHull(points);
		hull.build2D();
		PointND [] list = hull.getHullPoints(); 
		test.setColor(15);
		for (int i = 1; i < list.length; i++){
			test.drawLine((int)list[i-1].get(0), (int)list[i-1].get(1), (int)list[i].get(0), (int)list[i].get(1));
			test.putPixelValue((int)(list[i-1].get(0)), (int)(list[i-1].get(1)), 20.0);
			test.putPixelValue((int)(list[i-1].get(0)), (int)(list[i-1].get(1)), 20.0);
		}
		//test.drawLine((int)list[list.length-1].get(0), (int)list[list.length-1].get(1), (int)list[0].get(0), (int)list[0].get(0));
		//new ImageJ();
		//VisualizationUtil.showImageProcessor(test);
		for(int i = 0; i< points.length; i++){
			PointND p = points[i];
			if (!hull.isInside(p)){
				//System.out.println(p +  " " + i + " " + points.length);
			}
			Assert.assertEquals(true, hull.isInside(p));
		}
		Assert.assertEquals(false, hull.isInside(new Point3D(0,0,0)));
		Assert.assertEquals(false, hull.isInside(new Point3D(width,0,0)));
		Assert.assertEquals(false, hull.isInside(new Point3D(width,height,0)));
		Assert.assertEquals(false, hull.isInside(new Point3D(0,height,0)));
		//try {
		//	Thread.sleep(100000);
		//} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			//e.printStackTrace();
		//}
	}
	
	@Test
	public void testConvexHull3D(){
		int width = 200;
		int height = 200;
		int depth = 300;
		ImagePlus phantom = PhantomRenderer.createEmptyVolume("tes", width, height, depth);
		FloatProcessor test = (FloatProcessor) phantom.getStack().getProcessor(1);
		PointND [] points = new PointND[20000];
		for (int i = 0; i < points.length; i++){
			points[i] = new PointND(new SimpleVector(3));
			points[i].getAbstractVector().randomize(0, height*0.8);
			points[i].getAbstractVector().add(height*0.05);
			test = (FloatProcessor) phantom.getStack().getProcessor((int) (1+points[i].get(2)));
			test.putPixelValue((int)(points[i].get(0)), (int)(points[i].get(1)), 10.0);
		}
		ConvexHull hull = new ConvexHull(points);
		hull.build();
		PointND [] list = hull.getRasterPoints(2000); 
		test.setColor(15);
		for (int i = 0; i < list.length; i++){
			//System.out.println(list[i]);
			test = (FloatProcessor) phantom.getStack().getProcessor((int) Math.round(1+list[i].get(2)));
			test.putPixelValue((int)(list[i].get(0)), (int)(list[i].get(1)), 20.0);
		}
		//test.drawLine((int)list[list.length-1].get(0), (int)list[list.length-1].get(1), (int)list[0].get(0), (int)list[0].get(0));
		new ImageJ();
		phantom.show();
		//try {
		//	Thread.sleep(1000000);
		//} catch (InterruptedException e) {
			// TODO Auto-generated catch block
		//	e.printStackTrace();
		//}
		//VisualizationUtil.showImageProcessor(test);
		for(int i = 0; i< points.length; i++){
			PointND p = points[i];
			if (!hull.isInside(p)){
				System.out.println(p +  " " + i + " " + points.length);
			}
			Assert.assertEquals(true, hull.isInside(p));
		}
		Assert.assertEquals(false, hull.isInside(new Point3D(0,0,0)));
		Assert.assertEquals(false, hull.isInside(new Point3D(width,0,0)));
		Assert.assertEquals(false, hull.isInside(new Point3D(width,height,0)));
		Assert.assertEquals(false, hull.isInside(new Point3D(0,height,0)));

	}
	
	@Test
	public void testIsoCenterComputation(){
		CircularTrajectory geom = new CircularTrajectory(new Trajectory());
		geom.setSourceToDetectorDistance(1200);
		geom.setPixelDimensionX(0.616);
		geom.setPixelDimensionY(0.616);
		geom.setDetectorWidth(640);
		geom.setDetectorHeight(480);
		geom.setSourceToAxisDistance(800);
		geom.setTrajectory(200, 800, 1, 0, 0, Projection.CameraAxisDirection.DETECTORMOTION_PLUS, Projection.CameraAxisDirection.ROTATIONAXIS_PLUS, new SimpleVector(0,0,1));
		PointND isocenter = geom.computeIsoCenter();
		// Check consistency:
		ArrayList<PointND> testList1 = new ArrayList<PointND>();
		ArrayList<PointND> testList2 = new ArrayList<PointND>();
		for (int i = 0; i < 100; i++){
			testList1.add(geom.computeIsoCenterOld());
			testList2.add(geom.computeIsoCenter());
		}
		for (int i = 0; i < 100; i++){
			System.out.println(testList1.get(i));
			System.out.println(testList2.get(i));
		}
		System.out.println(DoublePrecisionPointUtil.getGeometricCenter(testList1) + " " + DoublePrecisionPointUtil.getGeometricCenter(testList2));
		System.out.println(DoublePrecisionPointUtil.getStandardDeviation(testList1) + " " + DoublePrecisionPointUtil.getStandardDeviation(testList2));
		// iso center new is way more stable:
		System.out.println(isocenter);
		Assert.assertEquals(new PointND(0,0,0), isocenter);
	}

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
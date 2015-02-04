package edu.stanford.rsl.conrad.geometry.splines;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;



public class SplineTests {

	public static BSpline createTestSpline(){
		double [] uVector = {0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1};
		
		ArrayList<PointND> list = new ArrayList<PointND>();
		list.add(new PointND (0, 0, 1));
		list.add(new PointND (1, 0, 2));
		list.add(new PointND (1, 0.75, 4));
		list.add(new PointND (2, 1, -1));
		list.add(new PointND (2, 0, 2));
		list.add(new PointND (2, 0.75, 4));
		list.add(new PointND (3, 1, -1));
		list.add(new PointND (3, 0, 2));
		list.add(new PointND (4, 1, -1));
		list.add(new PointND (4, 2, -1));
		list.add(new PointND (4, 0, 2));
		list.add(new PointND (5, 0.75, 4));
		list.add(new PointND (6, 1, -1));
		
		return new BSpline(list, uVector);
	}
	
	
	
	
	public static SurfaceBSpline createTestSurfaceSpline(){
		double [] uVector = {0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1, 1};
		double [] vVector = {0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1};
		
		ArrayList<PointND> list = new ArrayList<PointND>();
		list.add(new PointND (0, 0, 1));
		list.add(new PointND (1, 0, 2));
		list.add(new PointND (1, 0.75, 4));
		list.add(new PointND (2, 1, -1));
		list.add(new PointND (2, 0, 2));
		list.add(new PointND (2, 0.75, 4));
		list.add(new PointND (3, 1, -1));
		list.add(new PointND (3, 0, 2));
		list.add(new PointND (4, 1, -1));
		list.add(new PointND (4, 2, -1));
		list.add(new PointND (4, 0, 2));
		list.add(new PointND (5, 0.75, 4));
		list.add(new PointND (6, 1, -1));
		
		int i =1;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		
		i =3;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		
		i =5;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		
		i =9;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		
		i =11;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		i =12;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		i =13;
		list.add(new PointND (0, 0+i, 1));
		list.add(new PointND (1, 0+i, 2));
		list.add(new PointND (1, 0.75+i, 4));
		list.add(new PointND (2, 1+i, -1));
		list.add(new PointND (2, 0+i, 2));
		list.add(new PointND (2, 0.75+i, 4));
		list.add(new PointND (3, 1+i, -1));
		list.add(new PointND (3, 0+i, 2));
		list.add(new PointND (4, 1+i, -1));
		list.add(new PointND (4, 2+i, -1));
		list.add(new PointND (4, 0+i, 2));
		list.add(new PointND (5, 0.75+i, 4));
		list.add(new PointND (6, 1+i, -1));
		
		return new SurfaceBSpline("TestSpline", list, uVector, vVector);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// simple cubic spline

		BSpline spline = createTestSpline();
		UniformCubicBSpline cspline = new UniformCubicBSpline(spline.getControlPoints(), spline.getKnots());
		
		VisualizationUtil.createSplinePlot(spline).show();
		VisualizationUtil.createSplinePlot(cspline).show();
		
		FileReader file;
		try {
			file = new FileReader(FileUtil.myFileChoose(".nrb", false));
			SurfaceBSpline.readBSpline(new BufferedReader(file));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		
		
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
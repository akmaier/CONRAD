/*
 * Copyright (C) 2017 Andreas Maier, Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.dimreduction;


import java.io.IOException;

import edu.stanford.rsl.conrad.dimreduction.objfunctions.LagrangianDistanceObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.utils.MyException;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloud;
import edu.stanford.rsl.conrad.dimreduction.utils.PointCloudViewableOptimizableFunction;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SwissRoll;


public class Test {

	/**
	 * @param args
	 * @throws IOException
	 * @throws MyException
	 */
	public static void main(String[] args) throws IOException, MyException {
		
//		new GUI(); 
				
	SwissRoll sw = new SwissRoll(1.0, 10, 10); 
	PointCloud pc = new PointCloud(sw.getPointList()); 
	DimensionalityReduction dimRed = new DimensionalityReduction(pc); 
	PointCloudViewableOptimizableFunction gradFunc = new LagrangianDistanceObjectiveFunction(); 
	dimRed.setTargetFunction(gradFunc); 
	dimRed.setshowOrigPoints(true); 
	dimRed.setNormalOptimizationMode(true); 
	dimRed.setConvexityTest2D(true); 
	dimRed.set2Ddim(2); 
	dimRed.setFilenameConvexity2D(""); 
	dimRed.setConvexityTest2D(true); 
	dimRed.set3Ddim(0); 
	dimRed.setFilenameConvexity3D(""); 
	dimRed.setConfidenceMeasure(true); 
	dimRed.optimize();
		
		
//		Cube c = new Cube(1.0, 4, 0.01, 3, false, false, 0); 
//		PointCloud pc = new PointCloud(c.getPointList()); 
//		pc.normalizeInnerPointDistancesMean(1.0); 
//		DimensionalityReduction dimRed = new DimensionalityReduction(pc); 
//		PointCloudViewableOptimizableFunction gradFunc = new WeightedInnerProductObjectiveFunction(2, 0 ); 		
//		dimRed.setTargetFunction(gradFunc); 	
//		dimRed.setshowOrigPoints(false); 	
//		dimRed.setNormalOptimizationMode(false); 	
//		PlotParameterKError plotpke = new PlotParameterKError("Cube", 2, 3.0, 1.0 , "edgeLength" , 1.0 , 2.0, 0.5, "test123"); 	
//		dimRed.setPlotParameterKError(plotpke); 	
//		dimRed.optimize(); 
		
		
		
//		SwissRoll sw = new SwissRoll(1.0, 10, 10); 
//		PointCloud pc = new PointCloud(sw.getPointList()); 
//		DimensionalityReduction dimRed = new DimensionalityReduction(pc); 
//		PointCloudViewableOptimizableFunction gradFunc = new WeightedInnerProductObjectiveFunction(2, 0 ); 
//		dimRed.setTargetFunction(gradFunc); 
//		dimRed.setshowOrigPoints(false); 
//		dimRed.setNormalOptimizationMode(true); 
//		dimRed.setBestTimeValue(true); 
//		PlotIterationsError pie = new PlotIterationsError(); 
//		dimRed.setPlotIterError(pie);
//		((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).setK(2.0); 
//		PlotKError ke = new PlotKError(5.0, 1.0, true, 2, ""); 
//		dimRed.setPlotKError(ke); 
//		dimRed.optimize(); 
		
//		SwissRoll sw = new SwissRoll(1.0,30,5); 
//		PointCloud pc = new PointCloud(sw.getPointList());
//		DimensionalityReduction dimRed = new DimensionalityReduction(pc);
//		PointCloudViewableOptimizableFunction gradFunc = new SammonObjectiveFunction(); 
//		dimRed.setTargetFunction(gradFunc);
//		dimRed.setshowOrigPoints(true);
//		dimRed.setNormalOptimizationMode(true);
//		dimRed.setBestTimeValue(true);
//		PlotIterationsError pie = new PlotIterationsError();
//		dimRed.setPlotIterError(pie);
//		dimRed.optimize(); 
		
//		SwissRoll sw = new SwissRoll(1.0, 10, 10); 
//		PointCloud pc = new PointCloud(sw.getPointList()); 
//		DimensionalityReduction dimRed = new DimensionalityReduction(pc); 
//		PointCloudViewableOptimizableFunction gradFunc = new WeightedInnerProductObjectiveFunction(2, 0 ); 
//		dimRed.setTargetFunction(gradFunc); 
//		dimRed.setshowOrigPoints(false); 
//		dimRed.setNormalOptimizationMode(false);
//		PlotKError ke = new PlotKError(5.0, 0.5, true, 2, ""); 
//		dimRed.setPlotKError(ke); 
//		dimRed.optimize(); 
//		
		


	}

}

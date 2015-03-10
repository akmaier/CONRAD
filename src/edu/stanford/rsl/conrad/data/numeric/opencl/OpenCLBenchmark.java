/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.opencl;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

/**
 * OpenCLBenchmark is a tiny benchmark for testing local and global memory kernels and demo how an extension of new kernels works.
 */
public class OpenCLBenchmark {
	

	public static void main(String[] args) {
		
		CLContext context = OpenCLUtil.createContext();
		CLDevice[] devices = context.getDevices();	
		CLDevice device = devices[0];
		
		String leftAlignFormat = "| %-20s | %-5d | %-6d |%n";

		System.out.format("+----------------------+-------+--------+%n");
		System.out.printf("| size                 | local | global |%n");
		System.out.format("+----------------------+-------+--------+%n");
		
		int iterations = 10;
		
		for (int p = 2; p<12; p++)
		{
			int x = (int)Math.pow(2, p);
			int y = (int)Math.pow(2, p);
			
			Grid2D grid2D = new Grid2D(x, y);
			
			// fill the grid with 1.0f
			for(int i = 0; i<x; i++) {
				for(int j = 0; j<y; j++) {
					float val = 1.0f;
					grid2D.setAtIndex(i, j, val);
				}
			}
				
			
			// local version, using the standard OpenClGridOperators class
			
			OpenCLGrid2D openCLGrid2DLocal = new OpenCLGrid2D(grid2D, context, device);
			OpenCLGridOperators openCLGridOperatos = (OpenCLGridOperators) openCLGrid2DLocal.getGridOperator();
			openCLGridOperatos.sum(openCLGrid2DLocal); // initialize
			
			double sum = 0.0;
			
			long startLocal = System.currentTimeMillis();
			for (int i = 0; i<iterations; i++) 
			{
				sum = openCLGridOperatos.sum(openCLGrid2DLocal);
			}
			long endLocal = (System.currentTimeMillis() - startLocal);
			

			if ((int)sum != x*y) {
				System.out.println("\tWrong result in local sum: " + sum + " != " + x);
			}
			
			openCLGrid2DLocal.release();
			openCLGrid2DLocal = null;
			

			/*
			 * In this example we extend the standard OpenCL grid operators by a new method 'sumGlobalMemory(...)'. 
			 * Therefore we have to instantiate the new class ExtendedOpenCLGridOperators by its singleton.
			 * Then we can set this new, extended grid operator class to the grid.
			 */
			
			OpenCLGrid2D openCLGrid2DGlobal = new OpenCLGrid2D(grid2D, context, device);
			
			openCLGrid2DGlobal.setNumericGridOperator(ExtendedOpenCLGridOperators.getInstance()); // set a new numeric grid operator class
			
			ExtendedOpenCLGridOperators gg = (ExtendedOpenCLGridOperators) openCLGrid2DGlobal.getGridOperator(); // continue as usual, but with a cast
			gg.sumGlobalMemory(openCLGrid2DGlobal);
			
			
			sum = 0.0;

			long startGlobal = System.currentTimeMillis();
			for (int i = 0; i<iterations; i++) 
			{
				sum = gg.sumGlobalMemory(openCLGrid2DGlobal);
			}			
			long endGlobal = (System.currentTimeMillis() - startGlobal);
			
			if ( (int)sum != x*y ) {
				System.out.println("\tWrong result in global sum: " + sum + " != " + x*y);
			}
			
			openCLGrid2DGlobal.release();
			openCLGrid2DGlobal = null;
			
			
			System.out.format(leftAlignFormat, (x + " x " + y)+ " pixels ", endLocal/iterations, endGlobal/iterations);

				
		}
		
		System.out.format("+----------------------+-------+--------+%n");
		
		
		// last, but not lead: 
		// dot product
		Grid2D grid2D = new Grid2D(512, 512);
		for(int i = 0; i<512; i++) {
			for(int j = 0; j<512; j++) {
				float val = 1.0f;
				grid2D.setAtIndex(i, j, val);
			}
		}
		
		OpenCLGrid2D openCLGrid2D = new OpenCLGrid2D(grid2D, context, device);
		double dotProduct = openCLGrid2D.getGridOperator().dotProduct(openCLGrid2D, openCLGrid2D);
		System.out.println("dot product: " + dotProduct);

		//min
		openCLGrid2D.setAtIndex(1, 2, -0.12345f);
		float min = openCLGrid2D.getGridOperator().min(openCLGrid2D);
		System.out.println("min: " + min);
		

		
	}
}

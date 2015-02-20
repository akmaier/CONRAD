/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.opencl;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.opencl.delegates.OpenCLNumericMemoryDelegateLinear;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.tutorial.phantoms.Phantom;


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
			
			OpenCLGrid2D openCLGrid2D = new OpenCLGrid2D(grid2D, context, device);
			((OpenCLGridOperators) openCLGrid2D.getGridOperator()).sumGlobalMemory(openCLGrid2D); // initialize
			
			double sum = 0.0;

			long startGlobal = System.currentTimeMillis();
			for (int i = 0; i<iterations; i++) 
			{
				sum = ((OpenCLGridOperators) openCLGrid2D.getGridOperator()).sumGlobalMemory(openCLGrid2D);
			}			
			long endGlobal = (System.currentTimeMillis() - startGlobal);
			
			if ( (int)sum != x*y ) {
				System.out.println("\tWrong result in global sum: " + sum + " != " + x*y);
			}
			
			openCLGrid2D.release();
			openCLGrid2D = null;
			
			
			// local version
			
			OpenCLGrid2D openCLGrid2DLocal = new OpenCLGrid2D(grid2D, context, device);
			openCLGrid2DLocal.getGridOperator().sum(openCLGrid2DLocal); // initialize
			
			sum = 0.0;
			
			long startLocal = System.currentTimeMillis();
			for (int i = 0; i<iterations; i++) 
			{
				sum = openCLGrid2DLocal.getGridOperator().sum(openCLGrid2DLocal);
			}
			long endLocal = (System.currentTimeMillis() - startLocal);
			

			if ((int)sum != x*y) {
				System.out.println("\tWrong result in local sum: " + sum + " != " + x);
			}


			System.out.format(leftAlignFormat, (x + " x " + y)+ " pixels ", endLocal/iterations, endGlobal/iterations);
			
			openCLGrid2DLocal.release();
			openCLGrid2DLocal = null;
				
		}
		System.out.format("+----------------------+-------+--------+%n");
		
		
		//min
		Grid2D grid2D = new Grid2D(512, 512);
		
		grid2D.setAtIndex(1, 2, -0.12345f);
		
		OpenCLGrid2D openCLGrid2D = new OpenCLGrid2D(grid2D, context, device);
		openCLGrid2D.getGridOperator().sum(openCLGrid2D); // initialize
		
		float min = openCLGrid2D.getGridOperator().min(openCLGrid2D);
		System.out.println("min:" + min);
		

		
	}
}

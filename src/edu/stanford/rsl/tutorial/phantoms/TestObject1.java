package edu.stanford.rsl.tutorial.phantoms;

/**
 * Simple class to show the Grid2D functionality. We use a Grid2D to create a uniform circle.
 * We use this as a first phantom to investigate parallel beam projection and back-projection.
 * 
 * @author Recopra Summer Semester 2012
 *
 */
public class TestObject1 extends Phantom {

	/**
	 * The constructor takes two arguments to initialize the grid. The circle will be in the center and have a radius of 45% of the x-dimension.
	 * Thus we recommend a grid size that is uniform in both directions.
	 * @param x
	 * @param y
	 */
	public TestObject1 (int x, int y){
		super(x,y,"TestObject1");
		
		// Create circle in the grid.
		boolean ascending = true;
		int numCircles = 20;
		double angleStep = Math.PI*2 / (float) numCircles;
		double radius = 0.03*x;
		double bigRadius = 0.4*x;
		double val = 1.d;
		for (int i = 0; i < numCircles; i++){
//			if (i>= numCircles/2) continue;
			double angle = i * angleStep;
			int xCenter = (int) (x/2 + Math.cos(angle)*bigRadius);
			int yCenter = (int) (y/2 + Math.sin(angle)*bigRadius);
			for(int i1 = (int) (xCenter-radius*x -1); i1<xCenter+radius*x+1; i1++) {
				for(int j = 0; j<y; j++) {
					
					if( Math.pow(i1 - xCenter, 2)  + Math.pow(j - yCenter, 2) <= (radius*radius) ) {
						if (ascending) val = (float) i / (numCircles-1);
						super.setAtIndex(i1, j, (float) val);
					}	
				}
			}
		}
	
		
		
	}
	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
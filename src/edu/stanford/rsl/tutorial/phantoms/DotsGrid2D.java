package edu.stanford.rsl.tutorial.phantoms;

/**
 * Simple class to show the Grid2D functionality. We use a Grid2D to create a uniform circle.
 * We use this as a first phantom to investigate parallel beam projection and back-projection.
 * 
 * @author Recopra Summer Semester 2012
 *
 */
public class DotsGrid2D extends Phantom {

	/**
	 * The constructor takes two arguments to initialize the grid. The circles will be in the center and have a random radius and position.
	 * @param x
	 * @param y
	 */
	public DotsGrid2D (int x, int y){
		super(x,y,"DotsGrid2D");

		for(int num=0; num<3; ++num){
			// Create circle in the grid.
			double radius = (0.1+Math.random()/3)*x;
			int xcenter = x/((int)((0.2+Math.random())*5));
			int ycenter = y/((int)((0.2+Math.random())*5));
			float val = (float) Math.random();
			if(0.2 > val) val = 0.2f;

			for(int i = 0; i<x; i++) {
				for(int j = 0; j<y; j++) {
					if( Math.pow(i - xcenter, 2)  + Math.pow(j - ycenter, 2) <= (radius*radius) ) {
						super.setAtIndex(i, j, val);
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
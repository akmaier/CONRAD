package edu.stanford.rsl.tutorial.phantoms;

/**
 * Simple class to show the Grid2D functionality. We use a Grid2D to create a uniform circle.
 * We use this as a first phantom to investigate parallel beam projection and back-projection.
 * 
 * @author Recopra Summer Semester 2012
 *
 */
public class Sphere3D extends Phantom3D {

	/**
	 * The constructor takes two arguments to initialize the grid. The circles will be in the center and have a random radius and position.
	 * @param x
	 * @param y
	 */
	public Sphere3D (int x, int y,int z){
		super(x,y,z,"Sphere3D");

		double radius = 0.45 * x;
		int xcenter = x / 2;
		int ycenter = y / 2;
		int zcenter = z / 2;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				for (int k = 0; k < z; ++k){
				if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) + Math.pow(k - zcenter,2) <= (radius * radius)) {
					super.setAtIndex(i, j, k,val);
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
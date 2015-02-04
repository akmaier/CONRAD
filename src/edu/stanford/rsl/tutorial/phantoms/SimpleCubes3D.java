package edu.stanford.rsl.tutorial.phantoms;

/**
 * Simple class to show the Grid3D functionality. We use a Grid3D to create a cube containing 8 smaller cubes of different intensities.
 * We use this as a first phantom to investigate cone beam projection and back-projection.
 * 
 * @author Recopra Summer Semester 2012
 *
 */
public final class SimpleCubes3D extends Phantom3D {

	/**
	 * The constructor takes three arguments to initialize the grid. The cube will be in the center.
	 * @param x
	 * @param y
	 * @param z
	 */
	public SimpleCubes3D (int x, int y,int z){
		super(x, y, z, "SimpleCubes3D");
		
		final float VAL = 10.0f;
		
		for(int a=1; a<=8; ++a){
			final int start1 = (4 >= a)							? 0 : x/2;
			final int start2 = (0 == a % 4 || 0 == (a+1) % 4)	? 0 : y/2;
			final int start3 = (0 == a % 2)						? 0 : z/2;
			final int end1 = (4 >= a)							? x/2 : x;
			final int end2 = (0 == a % 4 || 0 == (a+1) % 4)		? y/2 : y;
			final int end3 = (0 == a % 2)						? z/2 : z;
//			System.out.println("START:\t" + start1 + "\t" + start2 + "\t" + start3);
//			System.out.println("END:\t" + end1 + "\t" + end2 + "\t" + end3);
			for (int i = start1; i < end1; i++) {
				for (int j = start2; j < end2; j++) {
					for (int k = start3; k < end3; ++k){
						super.setAtIndex(i, j, k, VAL + a);
//						System.out.println("super.setAtIndex(" + i + ", " + j + ", " + k + ", " + VAL * a + ");");
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

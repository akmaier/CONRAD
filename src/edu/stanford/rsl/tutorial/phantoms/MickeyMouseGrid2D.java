package edu.stanford.rsl.tutorial.phantoms;

public class MickeyMouseGrid2D extends Phantom{
	
	public MickeyMouseGrid2D(int x, int y) {
		this(x, y, 3);
	}
	
	public MickeyMouseGrid2D(int x, int y, int phase) {
		super(x,y,"MickeyMouseGrid2D");
		
		int val1 = 1;
		int val2 = 2;
		double r1 = 0.4*x;
		double r2 = 0.2*x;
		int xCenter1 = x/2;
		int yCenter1 = 6*y/10;
		
		int xCenter2 = (int) (0.25*x);
		int yCenter2 = (int) (0.15*y);
		int xCenter3 = (int) (0.75*x);
		int yCenter3 = (int) (0.15*y);
		
		for(int i = 0; i < x; i++){
			for(int j = 0; j < y; j++) {
			
				if (phase > 2) if( Math.pow(i - xCenter2, 2)  + Math.pow(j - yCenter2, 2) <= (r2*r2) ) {
					super.setAtIndex(i, j, val2);
				}
				if (phase > 1) if( Math.pow(i - xCenter3, 2)  + Math.pow(j - yCenter3, 2) <= (r2*r2) ) {
					super.setAtIndex(i, j, val2);
				}
				if (phase > 0) if( Math.pow(i - xCenter1, 2)  + Math.pow(j - yCenter1, 2) <= (r1*r1) ) {
					super.setAtIndex(i, j, val1);
				}
		}
	}

}}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
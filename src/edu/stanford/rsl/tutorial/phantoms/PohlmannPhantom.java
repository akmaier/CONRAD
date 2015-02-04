/*
 * Copyright (C) 2014 Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.phantoms;


/**
 * This class creates a custom phantom.
 *  
 * @author Marcel Pohlmann
 * 
 */

public class PohlmannPhantom extends Phantom{
	
	/**
	 * The constructor takes five arguments to initialize the grid. The circles will be in the center and have a radius 
	 * of [80%] (50%) of the parameter maxRadius [big circle] (small circle). On the circular curves are small circles with
	 * a radius of 10% of the parameter maxRadius, their intensities are in a descending order. 
	 * A grid size that is uniform in both directions is recommened.
	 * 
	 * @param x image size in x-direction
	 * @param y image size in y-direction
	 * @param numCircs number of circular aranged circles
	 * @param maxRadius maximum extend of the circular phantom
	 * @param water	set background intensities to 0.2
	 */
	public PohlmannPhantom(int x, int y, int numCircs, double maxRadius, boolean water){
		super(x, y);
		// Create circle in the grid.
		if(water){
			for(int i = 0; i < super.getHeight(); ++i){
				for(int j = 0; j < super.getWidth(); ++j){
					if((Math.pow(i - y/2, 2) + Math.pow(j - x/2, 2)) < (Math.pow(0.9*maxRadius, 2)))
						super.setAtIndex(j, i, 0.2f);
				}
			}
		}
		super.setAtIndex(x/2, y/2, 5.0f);
		boolean ascending = true;
		double angleStep = Math.PI*2 / (float) numCircs;
		double bigRadius = 0.8*maxRadius;
		double radius = 0.1*maxRadius;
		double val = 1.d;
		for (int i = 0; i < numCircs; i++){
			double angle = i * angleStep;
			int xCenter = (int) (x/2 + Math.cos(angle)*bigRadius);
			int yCenter = (int) (y/2 + Math.sin(angle)*bigRadius);
			for(int i1 = (int) (xCenter-radius*x -1); i1<xCenter+radius*x+1; i1++) {
				for(int j = 0; j<y; j++) {
					if( Math.pow(i1 - xCenter, 2)  + Math.pow(j - yCenter, 2) <= (radius*radius) ) {
						if (ascending) val = (float) i / (numCircs-1);
						super.setAtIndex(i1, j, (float) val);
					}
				}
			}
			xCenter = (int) (x/2 + Math.cos(angle)*0.5*bigRadius);
			yCenter = (int) (y/2 + Math.sin(angle)*0.5*bigRadius);
			for(int i1 = (int) (xCenter-radius*x -1); i1<xCenter+radius*x+1; i1++) {
				for(int j = 0; j<y; j++) {
					if( Math.pow(i1 - xCenter, 2)  + Math.pow(j - yCenter, 2) <= (0.5*radius*0.5*radius) ) {
						if (ascending) val = (1.0f - ((float) i / (numCircs-1)));
						super.setAtIndex(i1, j, (float) val);
					}
				}
			}

		}
	}

}

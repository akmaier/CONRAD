/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.apodization;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class WelchWindow extends ApodizationWindow {

	
	public static void main(String[] args){
		int width = 64;
		int height = 64;
		
		WelchWindow hw = new WelchWindow(width, height);
		Grid2D window = hw.getWindow();
		
		new ImageJ();
		window.show();	
	}
	
	public WelchWindow(int w, int h) {
		super(w, h);
	}

	@Override
	protected Grid2D setupWindow() {
		Grid2D window = new Grid2D(width, height);
		double termU = (width-1)/2.0d;
		double termV = (height-1)/2.0d;
		for(int i = 0; i < width; i++){
			double vU = ((i-termU) / termU);
			double valU = 1 - vU*vU;
			for(int j = 0; j < height; j++){
				double vV = ((j-termV) / termV);
				double valV = 1 - vV*vV;
				window.setAtIndex(i, j, (float)(valU*valV));
			}
		}
		return window;
	}

}

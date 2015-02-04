package edu.stanford.rsl.tutorial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.phantoms.DotsGrid2D;

/**
 * Simple example that diplays a uniform example.
 * @author Recopra Seminar Summer 2012
 *
 */
public class DisplayUniformCircle {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int imgSzXMM = 200,
			imgSzYMM = imgSzXMM;
		float pxSzXMM = 1.f,
			pxSzYMM = pxSzXMM;

		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM),
			imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM);
		new ImageJ();

		// phantom
		DotsGrid2D phantom = new DotsGrid2D(imgSzXGU, imgSzYGU);
		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU*phantom.getSpacing()[0]) / 2.0, -(imgSzYGU*phantom.getSpacing()[1]) / 2.0);
		Grid2D grid = phantom;

		grid.show("Uniform Circle Example");
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
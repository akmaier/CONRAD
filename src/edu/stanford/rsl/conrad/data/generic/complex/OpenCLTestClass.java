/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import ij.ImageJ;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;

public class OpenCLTestClass {
	public static void main(String[] args){
		Phantom grid = new UniformCircleGrid2D(512,512);
		ComplexGrid2D grid1 = new ComplexGrid2D(grid);
		grid1.activateCL();
		ComplexGrid2D grid2 = (ComplexGrid2D)grid1.clone();

		ComplexPointwiseOperators op = new ComplexPointwiseOperators();
		op.multiplyBy(grid1, grid2);

		new ImageJ();
		grid1.getRealGrid().show();
	}
}

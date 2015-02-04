package edu.stanford.rsl.conrad.data.test;

import org.junit.Assert;
import org.junit.Test;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class FFTTests {

	@Test
	public void testFFT1D(){
		float [] numbers = {1, 2, -4, 2, 1, 0};
		Grid1D grid = new Grid1D(numbers);
		Grid1DComplex complexGrid = new Grid1DComplex(grid);
		complexGrid.transformForward();
		complexGrid.transformInverse();
		NumericPointwiseOperators.divideBy(grid, -1);
		NumericGrid result = NumericPointwiseOperators.addedBy(grid, complexGrid.getRealSubGrid(0, numbers.length));
		float sum = (float)NumericPointwiseOperators.sum(result);
		Assert.assertTrue( Math.abs(sum)<CONRAD.FLOAT_EPSILON);
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
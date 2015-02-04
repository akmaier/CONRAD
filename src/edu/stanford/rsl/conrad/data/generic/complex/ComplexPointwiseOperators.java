/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;


import edu.stanford.rsl.conrad.data.generic.GenericPointwiseOperators;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;

public class ComplexPointwiseOperators extends GenericPointwiseOperators<Complex> {

	/** Build conjugate of a complex grid */
	public void conj(ComplexGrid grid) {
		grid.getGridOperator().conj(grid);
	}
	
	/** Build magnitude of a complex grid and return in real value / Imaginary part is 0*/
	public void abs(ComplexGrid grid){
		grid.getGridOperator().abs(grid);
	}
	
	/** Build magnitude of a complex grid and return in real value / Imaginary part is 0*/
	public ComplexGrid absCopy(ComplexGrid grid){
		ComplexGrid replaceGrid = (ComplexGrid)grid.clone();
		replaceGrid.getGridOperator().abs(replaceGrid);
		return replaceGrid;
	}
	
}

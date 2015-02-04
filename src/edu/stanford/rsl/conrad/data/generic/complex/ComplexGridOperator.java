/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;


import edu.stanford.rsl.conrad.data.generic.GenericGridOperator;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.generic.iterators.GenericPointwiseIteratorND;

public class ComplexGridOperator extends GenericGridOperator<Complex> implements ComplexGridOperatorInterface{
	
	public void conj(final ComplexGrid grid){
		GenericPointwiseIteratorND<Complex> it = new GenericPointwiseIteratorND<Complex>(grid);
		while(it.hasNext()){	
			it.setNext(new Complex(it.get().getReal(), -it.get().getImag()));
		}
	}
	
	public void abs(final ComplexGrid grid){
		GenericPointwiseIteratorND<Complex> it = new GenericPointwiseIteratorND<Complex>(grid);
		while(it.hasNext()){	
		    it.setNext(new Complex(it.get().getMagn(), 0));
		}
	}
	
	protected static ComplexGridOperator op = new ComplexGridOperator();
	
	public static ComplexGridOperator getInstance() {
		return op;
	}
}

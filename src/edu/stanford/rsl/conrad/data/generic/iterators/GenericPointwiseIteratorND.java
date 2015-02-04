/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.iterators;

import java.util.Iterator;

import edu.stanford.rsl.conrad.data.PointwiseIterator;
import edu.stanford.rsl.conrad.data.generic.GenericGrid;
import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;


public class GenericPointwiseIteratorND<T extends Gridable<T>> extends PointwiseIterator implements Iterator<T> {

	public GenericPointwiseIteratorND(GenericGrid<T> g) {
		grid = g;
		idx=new int[grid.getSize().length];
	}

	@Override
	public T next() {
		T val = ((GenericGrid<T>)grid).getValue(idx);
		iterate();
		return val;
	}

	public void setNext(T val) {
		((GenericGrid<T>)grid).setValue(val,idx);
		iterate();
	}
	
	public T get() {
		T val = ((GenericGrid<T>)grid).getValue(idx);
		return val;
	}

	public void set(T val) {
		((GenericGrid<T>)grid).setValue(val,idx);
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException("Grid iterators can not remove single elements from a grid");
	}
}

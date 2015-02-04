/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.iterators;

import edu.stanford.rsl.conrad.data.PointwiseIterator;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;


public class NumericPointwiseIteratorND extends PointwiseIterator {

	public NumericPointwiseIteratorND(NumericGrid g) {
		grid = g;
		idx = new int[g.getSize().length];
	}

	@Override
	public boolean hasNext() {
		for (int i = 0; i < grid.getSize().length; i++) {
			if (idx[i] >= grid.getSize()[i])
				return false;
		}
		return true;
	}

	public float getNext() {
		float val = ((NumericGrid)grid).getValue(idx);
		iterate();
		return val;
	}

	public void setNext(float val) {
		((NumericGrid)grid).setValue(val,idx);
		iterate();
	}
	
	public float get() {
		float val = ((NumericGrid)grid).getValue(idx);
		return val;
	}

	public void set(float val) {
		((NumericGrid)grid).setValue(val,idx);
	}

	public void iterate(){
		idx[0]++;
		for (int i = 0; i < grid.getSize().length-1; i++) {
			if (idx[i]>=grid.getSize()[i]){
				idx[i+1]++; idx[i]=0;
			}
		}
	}
}

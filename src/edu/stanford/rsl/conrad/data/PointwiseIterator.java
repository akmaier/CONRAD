/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data;




public abstract class PointwiseIterator{

	protected Grid grid;

	protected int dim;

	protected int[] idx;

	public boolean hasNext() {
		for (int i = 0; i < grid.getSize().length; i++) {
			if (idx[i] >= grid.getSize()[i])
				return false;
		}
		return true;
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



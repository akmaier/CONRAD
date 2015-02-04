/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic;

import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;

public interface GenericGridOperatorInterface<T extends Gridable<T>> {
	
	public void addBy(final GenericGrid<T> grid, T val);
	public void addBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB);
	public void subtractBy(final GenericGrid<T> grid, T val);
	public void subtractBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB);
	public void multiplyBy(final GenericGrid<T> grid, T val);
	public void multiplyBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB);
	public void divideBy(final GenericGrid<T> grid, T val);
	public void divideBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB);
	public void copy(final GenericGrid<T> gridA, final GenericGrid<T> gridB);
	public void fill(final GenericGrid<T> grid, T val);
	
	public T sum(final GenericGrid<T> grid);
	public T min(final GenericGrid<T> grid);
	public T max(final GenericGrid<T> grid);
	public T dotProduct(final GenericGrid<T> gridA, final GenericGrid<T> gridB);
}

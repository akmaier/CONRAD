/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic;

import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;
import edu.stanford.rsl.conrad.data.generic.iterators.GenericPointwiseIteratorND;


public abstract class GenericGridOperator<T extends Gridable<T>> implements GenericGridOperatorInterface<T>{
	
	/** Fill a GenericGrid<T> with the given value */
	public void fill(final GenericGrid<T> grid, T val) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		while (it.hasNext())
			it.setNext(val);
	}

	/** Get sum of all grid elements */
	public T sum(final GenericGrid<T> grid) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		T sum = (it.hasNext()) ? it.next() : null;
		while (it.hasNext())
			sum = it.next().add(sum);
		return sum;
	}

	/** Get min of a GenericGrid<T> */
	public T min(final GenericGrid<T> grid) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		T min = (it.hasNext()) ? it.next() : null;
		while (it.hasNext()) {
			min = ((it.get().compareTo(min)) < 0) ? it.get() : min;
			it.iterate();
		}
		return min;
	}

	/** Get max of a GenericGrid<T> */
	public T max(final GenericGrid<T> grid) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		T max = (it.hasNext()) ? it.next() : null;
		while (it.hasNext()) {
			max = ((it.get().compareTo(max)) > 0) ? it.get() : max;
			it.iterate();
		}
		return max;
	}

	/** Copy data of a GenericGrid<T> to another, not including boundaries */
	public void copy(GenericGrid<T> grid1, GenericGrid<T> grid2) {
		GenericPointwiseIteratorND<T> it1 = new GenericPointwiseIteratorND<T>(grid1);
		GenericPointwiseIteratorND<T> it2 = new GenericPointwiseIteratorND<T>(grid2);
		while (it1.hasNext() && it2.hasNext())
			it1.setNext(it2.next());
	}

	/** Compute dot product between grid1 and grid2 */
	public T dotProduct(GenericGrid<T> grid1, GenericGrid<T> grid2) {
		GenericPointwiseIteratorND<T> it1 = new GenericPointwiseIteratorND<T>(grid1);
		GenericPointwiseIteratorND<T> it2 = new GenericPointwiseIteratorND<T>(grid2);
		T value = (it1.hasNext() && it2.hasNext()) ? it1.next().mul(it2.next()) : null;
		while (it1.hasNext())
			value = value.add(it1.next().mul(it2.next()));
		return value;
	}

	/** Compute grid1 = grid1 - grid2 */
	public void addBy(GenericGrid<T> input, GenericGrid<T> add) {
		GenericPointwiseIteratorND<T> it_inout = new GenericPointwiseIteratorND<T>(input);
		GenericPointwiseIteratorND<T> it_add = new GenericPointwiseIteratorND<T>(add);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get().add(it_add.next()));
	}

	/** Compute grid = grid + a */
	public void addBy(GenericGrid<T> grid, T a) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		while (it.hasNext())
			it.setNext(it.get().add(a));
	}

	/** Compute grid1 = grid1 - grid2 */
	public void subtractBy(GenericGrid<T> input, GenericGrid<T> sub) {
		GenericPointwiseIteratorND<T> it_inout = new GenericPointwiseIteratorND<T>(input);
		GenericPointwiseIteratorND<T> it_sub = new GenericPointwiseIteratorND<T>(sub);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get().sub(it_sub.next()));
	}

	/** Compute grid = grid - a */
	public void subtractBy(GenericGrid<T> grid, T a) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		while (it.hasNext())
			it.setNext(it.get().sub(a));
	}

	public void divideBy(GenericGrid<T> input, GenericGrid<T> divisor) {
		GenericPointwiseIteratorND<T> it_inout = new GenericPointwiseIteratorND<T>(input);
		GenericPointwiseIteratorND<T> it_div = new GenericPointwiseIteratorND<T>(divisor);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get().div(it_div.next()));
	}

	/** Compute grid = grid / a */
	public void divideBy(GenericGrid<T> grid, T a) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		while (it.hasNext())
			it.setNext(it.get().div(a));
	}

	public void multiplyBy(GenericGrid<T> input, GenericGrid<T> multiplicator) {
		GenericPointwiseIteratorND<T> it_inout = new GenericPointwiseIteratorND<T>(input);
		GenericPointwiseIteratorND<T> it_mult = new GenericPointwiseIteratorND<T>(multiplicator);
		while (it_inout.hasNext() && it_mult.hasNext())
			it_inout.setNext(it_inout.get().mul(it_mult.next()));
	}

	/** Compute grid = grid * a */
	public void multiplyBy(GenericGrid<T> grid, T a) {
		GenericPointwiseIteratorND<T> it = new GenericPointwiseIteratorND<T>(grid);
		while (it.hasNext())
			it.setNext(it.get().mul(a));
	}
	
}

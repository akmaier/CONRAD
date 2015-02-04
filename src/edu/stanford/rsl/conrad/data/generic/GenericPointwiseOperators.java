/*
 * Copyright (C) 2010-2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic;

import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;


/** The collection of all operators working point-wise on GenericGrid<T> data. */
public abstract class GenericPointwiseOperators<T extends Gridable<T>>{
	
	/** Fill a GenericGrid<T> with the given value */
	public void fill(final GenericGrid<T> grid, T val) {
		grid.getGridOperator().fill(grid, val);
	}

	/** Get sum of all grid elements */
	public T sum(final GenericGrid<T> grid) {
		return grid.getGridOperator().sum(grid);
	}
	
	/** Get min of a GenericGrid<T> */
	public T min(final GenericGrid<T> grid) {
		return grid.getGridOperator().min(grid);
	}

	/** Get max of a GenericGrid<T> */
	public T max(final GenericGrid<T> grid) {
		return grid.getGridOperator().max(grid);
	}
	
	/** Copy data of a GenericGrid<T> to another, not including boundaries */
	public void copy(GenericGrid<T> grid1, GenericGrid<T> grid2) {
		GenericGridOperatorInterface<T> op = grid1.selectGridOperator(grid1, grid2);
		op.copy(grid1, grid2);
	}

	/** Compute dot product between grid1 and grid2 */
	public T dotProduct(GenericGrid<T> grid1, GenericGrid<T> grid2) {
		GenericGridOperatorInterface<T> op = grid1.selectGridOperator(grid1, grid2);
		return op.dotProduct(grid1, grid2);
	}

	/** Compute dot product between grid and itself. Same as square of l2 norm */
	public T dotProduct(GenericGrid<T> grid) {
		return grid.getGridOperator().dotProduct(grid, grid);
	}

	/** Compute grid3 = grid1 - grid2  */
	public GenericGrid<T> addedBy(GenericGrid<T> input, GenericGrid<T> sub) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, sub);
		GenericGrid<T> output=(GenericGrid<T>)input.clone();
		op.addBy(output, sub);
		return output; 
	}
	
	/** Compute grid1 = grid1 - grid2  */
	public void addBy(GenericGrid<T> input, GenericGrid<T> sub) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, sub);
		op.addBy(input, sub);
	}
	
	
	/** Compute grid = grid + a  */
	public void addBy(GenericGrid<T> grid, T a) {
		grid.getGridOperator().addBy(grid, a);
	}

	/** Compute grid3 = grid1 - grid2  */
	public GenericGrid<T> subtractedBy(GenericGrid<T> input, GenericGrid<T> sub) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, sub);
		GenericGrid<T> output = (GenericGrid<T>)input.clone();
		op.subtractBy(output, sub);
		return output;
	}
	
	/** Compute grid1 = grid1 - grid2  */
	public void subtractBy(GenericGrid<T> input, GenericGrid<T> sub) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, sub);
		op.subtractBy(input, sub);
	}
	
	
	
	/** Compute grid = grid - a  */
	public void subtractBy(GenericGrid<T> grid, T a) {
		grid.getGridOperator().subtractBy(grid, a);
	}
	
	public GenericGrid<T> dividedBy(GenericGrid<T> input, GenericGrid<T> divisor) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, divisor);
		GenericGrid<T> output = (GenericGrid<T>)input.clone();
		op.divideBy(output, divisor);
		return output;
	}
	
	public void divideBy(GenericGrid<T> input, GenericGrid<T> divisor) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, divisor);
		op.divideBy(input, divisor);
	}
	
	/** Compute grid = grid / a  */
	public void divideBy(GenericGrid<T> grid, T a) {
		grid.getGridOperator().divideBy(grid, a);
	}

	public GenericGrid<T> multipliedBy(GenericGrid<T> input, GenericGrid<T> multiplier) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, multiplier);
		GenericGrid<T> output = (GenericGrid<T>)input.clone();
		op.multiplyBy(output, multiplier);
		return output;
	}
	
	public void multiplyBy(GenericGrid<T> input, GenericGrid<T> multiplier) {
		GenericGridOperatorInterface<T> op = input.selectGridOperator(input, multiplier);
		op.multiplyBy(input, multiplier);
	}

	/** Compute grid = grid * a  */
	public void multiplyBy(GenericGrid<T> grid, T a) {
		grid.getGridOperator().multiplyBy(grid, a);
	}
}


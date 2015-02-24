/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;

/**
 * NumericGridOperator contains operations which can be applied to grids. It is implemented as singleton, because all grids share the same operations.
 */
public class NumericGridOperator {
	
	static NumericGridOperator op = new NumericGridOperator();
	protected NumericGridOperator() { }
	public static NumericGridOperator getInstance() {
		return op;
	}
	
	/** Fill a NumericGrid with the given value */
	public void fill(final NumericGrid grid, float val) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(val);
	}
	
	/** Fill a Grid's invalid elements with zero */
	public void fillInvalidValues(final NumericGrid grid) {
		fillInvalidValues(grid, 0);
	}
	
	/** Fill a Grid's invalid elements with the given value */
	public void fillInvalidValues(final NumericGrid grid, float val) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()){
			double gridVal = it.get();
			if (Double.isNaN(gridVal) || Double.isInfinite(gridVal))
				it.set(val);
			it.iterate();
		}
	}

	/** Get sum of all grid elements */
	public double sum(final NumericGrid grid) {
		double sum = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			sum += it.getNext();
		return (float) sum;
	}
	
	/** Get sum of all grid elements */
	public double sumSave(final NumericGrid grid) {
		double sum = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			double val = it.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val)))
				sum += val;
		}
		return (float) sum;
	}

	/** Get l1 norm of all grid elements */
	public double normL1(final NumericGrid grid) {
		double res = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			double val = it.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val))) {
				if (0 > val) // add abs
					res -= val;
				else
					res += val;
			}
		}
		return (float) res;
	}

	/** Get number of grid elements with negative values */
	public int countNegativeElements(final NumericGrid grid) {
		int res = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float val = it.getNext();
			if (!(Float.isInfinite(val) || Float.isNaN(val))) {
				if (0 > val) // add abs
					++res;
			}
		}
		return res;
	}

	public int countInvalidElements(NumericGrid grid) {
		int res = 0;
		
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float val = it.getNext();
			if (Float.isNaN(val) || Float.isInfinite(val))
				++res;
		}
		return res;
	}

	/** Get min of a NumericGrid */
	public float min(final NumericGrid grid) {
		float min = Float.MAX_VALUE;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid);
		while (it1.hasNext()) {
			if (it1.get() < min) {
				min = it1.get();
			}
			it1.getNext();
		}
		return min;
	}

	/** Get max of a NumericGrid */
	public float max(final NumericGrid grid) {
		float max = -Float.MAX_VALUE;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid);
		while (it1.hasNext()) {
			if (it1.get() > max)
				max = it1.get();
			it1.getNext();
		}
		return max;
	}

	/** Copy data of a NumericGrid to another, not including boundaries */
	public void copy(NumericGrid grid1, NumericGrid grid2) {
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext() && it2.hasNext())
			it1.setNext(it2.getNext());
	}

	/** Compute dot product between grid1 and grid2 */
	public double dotProduct(NumericGrid grid1, NumericGrid grid2) {
		double value = 0.0;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext())
			value += it1.getNext() * it2.getNext();
		return value;
	}
	
	/** Compute weighted dot product between grid1 and grid2 */
	public double weightedDotProduct(NumericGrid grid1, NumericGrid grid2, double weightGrid2, double addGrid2) {
		double value = 0.0;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext())
			value += it1.getNext() * (it2.getNext()*weightGrid2 + addGrid2);
		return value;
	}
	
	/** Compute dot product between grid1 and grid2 */
	public double weightedSSD(NumericGrid grid1, NumericGrid grid2, double weightGrid2, double addGrid2) {
		double value = 0.0;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext()){
			double val = (it1.getNext() - (it2.getNext()*weightGrid2 + addGrid2));
			value += val*val;
		}
		return value;
	}
	
	/** Compute rmse between grid1 and grid2 */
	public double rmse(NumericGrid grid1, NumericGrid grid2) {
		double sum = 0.0;
		long numErrors = 0;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext() && it2.hasNext()) {
			double val = it1.getNext() - it2.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val)))
				sum += val * val;
			else
				++numErrors;
		}
		if (0 != numErrors)
			System.err.println("Errors in RMSE computation: "
					+ ((double) numErrors * 100)
					/ (grid1.getNumberOfElements()) + "%");
		return Math.sqrt(sum/grid1.getNumberOfElements());
	}
	
	/** Compute grid1 = grid1 + grid2 */
	public void addBySave(NumericGrid input, NumericGrid sum) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sum = new NumericPointwiseIteratorND(sum);
		while (it_inout.hasNext()) {
			double b = it_inout.get();
			double a = it_sum.getNext();
			if ((Double.isInfinite(b) || Double.isNaN(b)))
				b = 0;
			if(Double.isInfinite(a) || Double.isNaN(a))
				a = 0;
			it_inout.setNext((float) (a + b));
		}
	}

	/** Compute grid1 = grid1 - grid2 */
	public void addBy(NumericGrid input, NumericGrid sub) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sub = new NumericPointwiseIteratorND(sub);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get() + it_sub.getNext());
	}

	/** Compute grid = grid + a */
	public void addBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() + a);
	}
	
	/** Compute grid1 = grid1 - grid2 */
	public void subtractBySave(NumericGrid input, NumericGrid sub) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sum = new NumericPointwiseIteratorND(sub);
		while (it_inout.hasNext()) {
			double b = it_inout.get();
			double a = it_sum.getNext();
			if ((Double.isInfinite(b) || Double.isNaN(b)))
				b = 0;
			if(Double.isInfinite(a) || Double.isNaN(a))
				a = 0;
			it_inout.setNext((float) (a - b));
		}
	}

	/** Compute grid1 = grid1 - grid2 */
	public void subtractBy(NumericGrid input, NumericGrid sub) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sub = new NumericPointwiseIteratorND(sub);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get() - it_sub.getNext());
	}

	/** Compute grid = grid - a */
	public void subtractBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() - a);
	}
	
	/**
	 * Compute grid = grid * a in case of NaN or infinity 0 is set
	 */
	public void divideBySave(NumericGrid input, NumericGrid divisor) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_div = new NumericPointwiseIteratorND(divisor);
		while (it_inout.hasNext() && it_div.hasNext()) {
			double a = it_inout.get();
			double b = it_div.getNext();
			if (0 == a || 0 == b || Double.isInfinite(b) || Double.isNaN(b) || Double.isInfinite(a) || Double.isNaN(a))
				it_inout.setNext(0);
			else
				it_inout.setNext((float) (a / b));
		}
	}

	public void divideBy(NumericGrid input, NumericGrid divisor) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_div = new NumericPointwiseIteratorND(divisor);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get() / it_div.getNext());
	}

	/** Compute grid = grid / a */
	public void divideBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() / a);
	}

	public void multiplyBy(NumericGrid input, NumericGrid multiplicator) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_mult = new NumericPointwiseIteratorND(multiplicator);
		while (it_inout.hasNext() && it_mult.hasNext())
			it_inout.setNext(it_inout.get() * it_mult.getNext());
	}
	
	/**
	 * Compute grid = grid * a in case of nan or infty 0 is set
	 */
	public void multiplyBySave(NumericGrid input, NumericGrid multiplicator) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_mult = new NumericPointwiseIteratorND(multiplicator);
		while (it_inout.hasNext() && it_mult.hasNext()) {
			double a = it_inout.get();
			double b = it_mult.getNext();
			if ((Double.isInfinite(b) || Double.isNaN(b)
					|| Double.isInfinite(a) || Double.isNaN(a)))
				it_inout.setNext(0);
			else
				it_inout.setNext((float) (a * b));
		}
	}

	/** Compute grid = grid * a */
	public void multiplyBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() * a);
	}
	
	/**
	 * Compute grid = grid * a in case of NaN or infinity 0 is set
	 */
	public void multiplyBySave(NumericGrid grid, float b) {
		if(Double.isInfinite(b) || Double.isNaN(b))
			System.err.println("[multiplyBySave] called with invalid scalar value");
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			double a = it.get();
			if (Double.isInfinite(a) || Double.isNaN(a))
				it.setNext(0);
			else
				it.setNext((float)(a * b));
		}
	}

	/** Set all negative values in grid as zero. */
	public void removeNegative(NumericGrid grid) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext((it.get() < 0) ? 0 : it.get());
	}

	public double stddev(NumericGrid data, double mean) {
		double theStdDev = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext()){
			double value =(it.getNext() - mean);
			theStdDev += value*value;
		}
		return Math.sqrt(theStdDev / data.getNumberOfElements());
	}

	public void abs(NumericGrid data) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext(Math.abs(it.get()));
	}

	public void pow(NumericGrid grid, double exponent) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext((float) Math.pow(it.get(), (float)exponent));
	}
	
	public void sqrt(NumericGrid grid) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext((float) Math.sqrt(it.get()));
	}

	public void log(NumericGrid data) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.log(it.get()));
	}

	public void exp(NumericGrid data) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.exp(it.get()));
	}
	
	/** set maximum value, all values > max are set to max */
	public void setMax(NumericGrid data, float max) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.min(max, it.get()));
	}

	/** set minimum value, all values < min are set to min */
	public void setMin(NumericGrid data, float min) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.max(min, it.get()));
	}
	
	/* transpose grid - Caution, not optimized yet */
	public Grid2D transpose(Grid2D grid) {
		Grid2D gridT = new Grid2D(grid.getSize()[1], grid.getSize()[0]);
		for(int i=0; i<grid.getSize()[0]; ++i)
			for(int j=0; j<grid.getSize()[1]; ++j)
				gridT.addAtIndex(j, i, grid.getAtIndex(i, j));
		double norm1 = normL1(grid);
		double norm2 = normL1(gridT);
		if(norm1 != norm2)
			System.err.println("Error in transpose");
		return gridT;
	}
}

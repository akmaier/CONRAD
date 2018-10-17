/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;

public class NumericGridOperator {

	static NumericGridOperator op = new NumericGridOperator();

	public static NumericGridOperator getInstance() {
		return op;
	}

	/** Fill a NumericGrid with the given value */
	public void fill(NumericGrid grid, float val) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(val);
		grid.notifyAfterWrite();
	}

	/** Fill a Grid's invalid elements with zero */
	public void fillInvalidValues(final NumericGrid grid) {
		fillInvalidValues(grid, 0);
	}

	/** Fill a Grid's invalid elements with the given value */
	public void fillInvalidValues(NumericGrid grid, float val) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float gridVal = it.get();
			if (Double.isNaN(gridVal) || Double.isInfinite(gridVal))
				it.set(val);
			it.iterate();
		}
		grid.notifyAfterWrite();
	}

	/** Get sum of all grid elements */
	public float sum(final NumericGrid grid) {
		float sum = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			sum += it.getNext();
		return (float) sum;
	}

	/** Get sum of all grid elements. Ignores nans and infinity */
	public float sumSave(final NumericGrid grid) {
		float sum = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float val = it.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val)))
				sum += val;
		}
		return (float) sum;
	}

	/** Get l1 norm of all grid elements */
	public float normL1(final NumericGrid grid) {
		double res = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float val = it.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val))) {
				if (0 > val) // add abs
					res -= val;
				else
					res += val;
			}
		}
		return (float) res;
	}

	/** Get l2 norm of all grid elements */
	public float normL2(final NumericGrid grid) {
		double res = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float val = it.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val))) {
				res += Math.pow(val, 2);
			}
		}
		return (float) Math.sqrt(res);
	}

	/**
	 * Get number of grid elements with negative values but it doesnt count if
	 * negative infinity?!
	 */
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

	public int countInvalidElements(final NumericGrid grid) {
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

	/**
	 * Copy data of a NumericGrid to another, not including boundaries.
	 * Overwrites grid1
	 */
	public void copy(NumericGrid grid1, NumericGrid grid2) {
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext() && it2.hasNext())
			it1.setNext(it2.getNext());
		grid1.notifyAfterWrite();
	}

	/** Compute dot product between grid1 and grid2 */
	public float dotProduct(final NumericGrid grid1, NumericGrid grid2) {
		float value = 0.0f;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext())
			value += it1.getNext() * it2.getNext();
		return value;
	}

	/** Compute weighted dot product between grid1 and grid2 */
	public float weightedDotProduct(final NumericGrid grid1, final NumericGrid grid2, float weightGrid2,
			float addGrid2) {
		float value = 0.0f;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext())
			value += it1.getNext() * (it2.getNext() * weightGrid2 + addGrid2);
		return value;
	}

	/** Compute dot product between grid1 and grid2 */
	public float weightedSSD(final NumericGrid grid1, final NumericGrid grid2, double weightGrid2, double addGrid2) {
		float value = 0.0f;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext()) {
			float val = (float) (it1.getNext() - (it2.getNext() * weightGrid2 + addGrid2));
			value += val * val;
		}
		return value;
	}

	/** Compute rmse between grid1 and grid2 */
	public float rmse(final NumericGrid grid1, final NumericGrid grid2) {
		float sum = 0.0f;
		long numErrors = 0;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext() && it2.hasNext()) {
			float val = it1.getNext() - it2.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val)))
				sum += val * val;
			else
				++numErrors;
		}
		if (0 != numErrors)
			System.err.println(
					"Errors in RMSE computation: " + ((float) numErrors * 100) / (grid1.getNumberOfElements()) + "%");
		return (float) Math.sqrt(sum / grid1.getNumberOfElements());
	}

	/** Compute grid1 = grid1 + grid2. Ignores nans and infinity */
	public void addBySave(NumericGrid input, NumericGrid sum) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sum = new NumericPointwiseIteratorND(sum);
		while (it_inout.hasNext()) {
			float b = it_inout.get();
			float a = it_sum.getNext();
			if ((Double.isInfinite(b) || Double.isNaN(b)))
				b = 0;
			if (Double.isInfinite(a) || Double.isNaN(a))
				a = 0;
			it_inout.setNext((float) (a + b));
		}
		input.notifyAfterWrite();
	}

	/** Compute grid1 = grid1 - grid2 */
	public void addBy(NumericGrid input, NumericGrid sub) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sub = new NumericPointwiseIteratorND(sub);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get() + it_sub.getNext());
		input.notifyAfterWrite();
	}

	/** Compute grid = grid + a */
	public void addBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() + a);
		grid.notifyAfterWrite();
	}

	/** Compute grid = grid + a. Ignores nans and infinity */
	public void addBySave(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float nextValue = it.get();
			if (Double.isInfinite(nextValue) || Double.isNaN(nextValue))
				nextValue = 0;
			it.setNext(nextValue + a);
		}
		grid.notifyAfterWrite();
	}

	/** Compute grid1 = grid1 - grid2. Ignores nans and infinity */
	public void subtractBySave(NumericGrid input, NumericGrid sub) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sum = new NumericPointwiseIteratorND(sub);
		while (it_inout.hasNext()) {
			float b = it_inout.get();
			float a = it_sum.getNext();
			if ((Double.isInfinite(b) || Double.isNaN(b)))
				b = 0;
			if (Double.isInfinite(a) || Double.isNaN(a))
				a = 0;
			it_inout.setNext((float) (a - b));
		}
		input.notifyAfterWrite();
	}

	/** Compute grid = grid - a. Ignores nans and infinity */
	public void subtractBySave(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float nextValue = it.get();
			if (Double.isInfinite(nextValue) || Double.isNaN(nextValue))
				nextValue = 0;
			it.setNext(nextValue - a);
		}
		grid.notifyAfterWrite();
	}

	/** Compute grid1 = grid1 - grid2 */
	public void subtractBy(NumericGrid input, NumericGrid sub) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_sub = new NumericPointwiseIteratorND(sub);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get() - it_sub.getNext());
		input.notifyAfterWrite();
	}

	/** Compute grid = grid - a */
	public void subtractBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() - a);
		grid.notifyAfterWrite();
	}

	/**
	 * Compute grid = grid * a in case of NaN or infinity 0 is set
	 */
	public void divideBySave(NumericGrid input, NumericGrid divisor) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_div = new NumericPointwiseIteratorND(divisor);
		while (it_inout.hasNext() && it_div.hasNext()) {
			float a = it_inout.get();
			float b = it_div.getNext();
			if (0 == a || 0 == b || Double.isInfinite(b) || Double.isNaN(b) || Double.isInfinite(a) || Double.isNaN(a))
				it_inout.setNext(0);
			else
				it_inout.setNext((float) (a / b));
		}
		input.notifyAfterWrite();
	}

	public void divideBy(NumericGrid input, NumericGrid divisor) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_div = new NumericPointwiseIteratorND(divisor);
		while (it_inout.hasNext())
			it_inout.setNext(it_inout.get() / it_div.getNext());
		input.notifyAfterWrite();
	}

	/** Compute grid = grid / a */
	public void divideBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() / a);
		grid.notifyAfterWrite();
	}

	/** Compute grid = grid / a. Ignores nans and infinity */
	public void divideBySave(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float nextValue = it.get();
			if (Double.isInfinite(nextValue) || Double.isNaN(nextValue))
				nextValue = 0;
			it.setNext(nextValue / a);
		}
		grid.notifyAfterWrite();
	}

	public void multiplyBy(NumericGrid input, NumericGrid multiplicator) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_mult = new NumericPointwiseIteratorND(multiplicator);
		while (it_inout.hasNext() && it_mult.hasNext())
			it_inout.setNext(it_inout.get() * it_mult.getNext());
		input.notifyAfterWrite();
	}

	/**
	 * Compute grid = grid * a in case of nan or infty 0 is set
	 */
	public void multiplyBySave(NumericGrid input, NumericGrid multiplicator) {
		NumericPointwiseIteratorND it_inout = new NumericPointwiseIteratorND(input);
		NumericPointwiseIteratorND it_mult = new NumericPointwiseIteratorND(multiplicator);
		while (it_inout.hasNext() && it_mult.hasNext()) {
			float a = it_inout.get();
			float b = it_mult.getNext();
			if ((Double.isInfinite(b) || Double.isNaN(b) || Double.isInfinite(a) || Double.isNaN(a)))
				it_inout.setNext(0);
			else
				it_inout.setNext((float) (a * b));
		}
		input.notifyAfterWrite();
	}

	/** Compute grid = grid * a */
	public void multiplyBy(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext(it.get() * a);
		grid.notifyAfterWrite();
	}

	/**
	 * Compute grid = grid * a in case of NaN or infinity 0 is set
	 */
	public void multiplyBySave(NumericGrid grid, float b) {
		if (Double.isInfinite(b) || Double.isNaN(b))
			System.err.println("[multiplyBySave] called with invalid scalar value");
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float a = it.get();
			if (Double.isInfinite(a) || Double.isNaN(a))
				it.setNext(0);
			else
				it.setNext((float) (a * b));
		}
		grid.notifyAfterWrite();
	}

	/** Set all negative values in grid as zero. */
	public void removeNegative(NumericGrid grid) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext((it.get() < 0) ? 0 : it.get());
		grid.notifyAfterWrite();
	}

	public float stddev(final NumericGrid data, double mean) {
		float theStdDev = 0;
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext()) {
			float value = (float) (it.getNext() - mean);
			theStdDev += value * value;
		}
		return (float) Math.sqrt(theStdDev / data.getNumberOfElements());
	}

	public void abs(NumericGrid data) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext(Math.abs(it.get()));
		data.notifyAfterWrite();
	}

	public void pow(NumericGrid grid, double exponent) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext((float) Math.pow(it.get(), (float) exponent));
		grid.notifyAfterWrite();
	}

	public void sqrt(NumericGrid grid) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext())
			it.setNext((float) Math.sqrt(it.get()));
		grid.notifyAfterWrite();
	}

	public void log(NumericGrid data) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.log(it.get()));
		data.notifyAfterWrite();
	}

	public void exp(NumericGrid data) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.exp(it.get()));
		data.notifyAfterWrite();
	}

	/** set maximum value, all values > max are set to max */
	public void setMax(NumericGrid data, float max) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.min(max, it.get()));
		data.notifyAfterWrite();
	}

	/** set minimum value, all values < min are set to min */
	public void setMin(NumericGrid data, float min) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(data);
		while (it.hasNext())
			it.setNext((float) Math.max(min, it.get()));
		data.notifyAfterWrite();
	}

	/* transpose grid - Caution, not optimized yet */
	public Grid2D transpose(Grid2D grid) {
		Grid2D gridT = new Grid2D(grid.getSize()[1], grid.getSize()[0]);
		for (int i = 0; i < grid.getSize()[0]; ++i)
			for (int j = 0; j < grid.getSize()[1]; ++j)
				gridT.addAtIndex(j, i, grid.getAtIndex(i, j));
		float norm1 = normL1(grid);
		float norm2 = normL1(gridT);
		if (norm1 != norm2)
			System.err.println("Error in transpose");
		return gridT;
	}

	/* convert grid2d[] to grid3d */
	public void convert2DArrayTo3D(NumericGrid gridRes, final NumericGrid[] grid) {
		if (grid.length != gridRes.getSize()[2] || grid[0].getSize()[1] != gridRes.getSize()[1]
				|| grid[0].getSize()[0] != gridRes.getSize()[0])
			System.err.println("Sizes dont match");
		else {
			for (int z = 0; z < grid.length; z++)
				for (int y = 0; y < grid.length; y++)
					for (int x = 0; x < grid.length; x++)
						gridRes.setValue(grid[z].getValue(new int[] { x, y }), new int[] { x, y, z });
			gridRes.notifyAfterWrite();
		}
	}

	/**
	 * subtract of two grids with given offset
	 * 
	 * @param gridRes
	 *            = result grid
	 * @param gridA,
	 *            gridB = input grids
	 * @param x,y,z
	 *            Offset = offsetvalue in x,y,and z direction
	 * @param offsetleft
	 *            = true if left offset, false if right offset
	 */
	public void subtractOffset(NumericGrid gridResult, final NumericGrid gridA, final NumericGrid gridB, int xOffset,
			int yOffset, int zOffset, boolean offsetleft) {

		if (gridA.getSize()[0] != gridB.getSize()[0] || gridA.getSize()[1] != gridB.getSize()[1]
				|| gridA.getSize()[2] != gridB.getSize()[2])
			System.err.println("Grids have different sizes so they can not be subtracted.");

		for (int x = xOffset; x < gridA.getSize()[0] + xOffset; ++x)
			for (int y = yOffset; y < gridA.getSize()[1] + yOffset; ++y)
				for (int z = zOffset; z < gridA.getSize()[2] + zOffset; ++z) {

					int xIdx = (x >= gridA.getSize()[0] || x < 0) ? Math.min(Math.max(0, x), gridA.getSize()[0] - 1)
							: x;
					int yIdx = (y >= gridA.getSize()[1] || y < 0) ? Math.min(Math.max(0, y), gridA.getSize()[1] - 1)
							: y;
					int zIdx = (z >= gridA.getSize()[2] || z < 0) ? Math.min(Math.max(0, z), gridA.getSize()[2] - 1)
							: z;

					if (offsetleft)
						gridResult.setValue(
								gridA.getValue(new int[] { xIdx, yIdx, zIdx })
										- gridB.getValue(new int[] { x - xOffset, y - yOffset, z - zOffset }),
								new int[] { x - xOffset, y - yOffset, z - zOffset });
					else
						gridResult.setValue(
								gridA.getValue(new int[] { x - xOffset, y - yOffset, z - zOffset })
										- gridB.getValue(new int[] { xIdx, yIdx, zIdx }),
								new int[] { x - xOffset, y - yOffset, z - zOffset });
				}
		gridResult.notifyAfterWrite();
	}

	/**
	 * gradient in x-,y- or z-direction
	 * 
	 * @param gridRes
	 *            = result grid
	 * @param grid
	 *            = input grid
	 * @param value
	 *            = offsetvalue
	 * @param offsetleft
	 *            = true if left offset, false if right offset
	 */
	public void gradX(NumericGrid gridRes, final NumericGrid grid, int value, boolean offsetleft) {
		subtractOffset(gridRes, grid, grid, value, 0, 0, offsetleft);
		gridRes.notifyAfterWrite();
	}

	public void gradY(NumericGrid gridRes, final NumericGrid grid, int value, boolean offsetleft) {
		subtractOffset(gridRes, grid, grid, 0, value, 0, offsetleft);
		gridRes.notifyAfterWrite();
	}

	public void gradZ(NumericGrid gridRes, final NumericGrid grid, int value, boolean offsetleft) {
		subtractOffset(gridRes, grid, grid, 0, 0, value, offsetleft);
		gridRes.notifyAfterWrite();
	}

	/**
	 * calculates the divergence in x-,y-, or z-direction
	 * 
	 * @param gridRes
	 *            = result grid
	 * @param grid
	 *            = input grid
	 * @param x,y,z
	 *            Offset = offsetvalue in x,y,and z direction
	 * @param offsetleft
	 *            = true if left offset, false if right offset
	 */
	public void divergence(NumericGrid gridRes, final NumericGrid grid, int xOffset, int yOffset, int zOffset,
			boolean offsetleft) {

		if (xOffset == 0 && yOffset == 0 && zOffset == 0)
			System.err.println("No offset value chosen");
		else if ((xOffset != 0 && (yOffset != 0 || zOffset != 0)) || (yOffset != 0 && zOffset != 0))
			System.err.println("Too many divergence offsets chosen");
		else {

			// x:0 y:1 z:2
			int mode = 0;
			if (xOffset != 0) {
				gridRes.getGridOperator().gradX(gridRes, grid, xOffset, offsetleft);
				mode = 0;
			} else if (yOffset != 0) {
				gridRes.getGridOperator().gradY(gridRes, grid, yOffset, offsetleft);
				mode = 1;
			} else {
				gridRes.getGridOperator().gradZ(gridRes, grid, zOffset, offsetleft);
				mode = 2;
			}
			int sizeE = gridRes.getSize()[0];
			int sizeF = gridRes.getSize()[1];

			for (int e = 0; e < sizeE; ++e)
				for (int f = 0; f < sizeF; ++f) {
					if (mode == 0) {
						gridRes.setValue(-grid.getValue(new int[] { gridRes.getSize()[0] - 2, e, f }),
								new int[] { gridRes.getSize()[0] - 1, e, f });
					} else if (mode == 1) {
						gridRes.setValue(grid.getValue(new int[] { e, 0, f }), new int[] { e, 0, f });
						gridRes.setValue(-grid.getValue(new int[] { e, gridRes.getSize()[1] - 2, f }),
								new int[] { e, gridRes.getSize()[1] - 1, f });
					} else {
						gridRes.setValue(grid.getValue(new int[] { e, f, 0 }), new int[] { e, f, 0 });
						gridRes.setValue(-grid.getValue(new int[] { e, f, gridRes.getSize()[2] - 2 }),
								new int[] { e, f, gridRes.getSize()[2] - 1 });
					}
				}
		}
	}

	// Extension of original class by Christopher Fichtel based on class
	// Arithmetics
	// More numeric operations (also in specific region of interest roi)

	/** Lets the iterator go to the given index */
	public void goToIndex(final NumericPointwiseIteratorND it, long idx) {
		int counter = 0;
		while (counter < idx) {
			if (it.hasNext()) {
				it.iterate();
				counter++;
			} else {
				System.err.println("Submitted index or point is not in value range.");
				return;
			}
		}
	}

	/**
	 * gets the index of a point committed as int[]; works with dimensions <= 3
	 */
	public long getIndex(final NumericGrid grid, int[] point) {
		int[] size = grid.getSize();
		long idx = 0;
		if (size.length != point.length) {
			System.err.println("Dimensions of starting point and image don't match at method 'getIndex'!");
		} else {
			for (int i = point.length - 1; i >= 0; i--) {
				int a = (i - 2 >= 0) ? size[i - 2] : 1;
				int b = (i - 1 >= 0) ? size[i - 1] : 1;
				idx += (a * b) * point[i];
			}
		}
		return idx;
	}

	/** gets value at a certain point commited as int array; */
	public float getValAtPoint(final NumericGrid grid, int[] point) {
		long idx = op.getIndex(grid, point);
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		op.goToIndex(it, idx);
		float val = it.get();
		if (!(Double.isInfinite(val) || Double.isNaN(val)))
			return val;
		else
			System.err.println("Value at submitted index is NaN or infinite.");
		return 0;
	}

	/**
	 * Gets sum and squared sum of all grid elements. Ignores nans and infinity
	 */
	public double[] computeSumSquaredSum(final NumericGrid grid) {
		double sum = 0;
		double squaredSum = 0;

		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float val = it.getNext();
			if (!(Double.isInfinite(val) || Double.isNaN(val))) {
				sum += val;
				squaredSum += val * val;
			} else {
				System.err.println("Grid contains invalid values NaN or Infinite.");
			}
		}
		return new double[] { sum, squaredSum };
	}

	/**
	 * Gets sum and squared sum of all grid elements in given roi. Ignores nans
	 * and infinity
	 */
	public double[] computeSumSquaredSum(final NumericGrid grid, final NumericGrid roi) {
		int[] indices = new int[0];
		return (op.computeSumSquaredSum(grid, roi, indices));
	}

	/**
	 * Gets sum and squared sum of all grid elements in given roi (region of
	 * interest). Ignores NaNs or infinity; (dimension of roi) < (dimension of
	 * grid), therefore an index-array is submitted to create subgrids
	 */
	public double[] computeSumSquaredSum(final NumericGrid grid, final NumericGrid roi, int[] indices) {
		int[] sizeGrid = grid.getSize();
		int[] sizeRoi = roi.getSize();
		int difference = sizeGrid.length - sizeRoi.length;
		NumericGrid clone = grid.clone();
		double sum = 0;
		double squaredSum = 0;
		int numErrors = 0;
		if (difference != indices.length)
			System.err.println("Error at computation of SumSquaredSum. Submitted Array of indices should have length "
					+ difference);

		// create subgrids of original image in order to match dimension of roi
		for (int i = 0; i < indices.length; i++) {
			if (indices[i] < 0 || indices[i] > sizeGrid[sizeGrid.length - 1] - 1) {
				System.err.println("Invalid Index. Index at position " + i + " has to be in intervall [0, "
						+ (sizeGrid[sizeGrid.length - 1] - 1) + "]");
			}
			clone = clone.getSubGrid(indices[i]);
		}
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(clone);
		NumericPointwiseIteratorND itRoi = new NumericPointwiseIteratorND(roi);
		while (it.hasNext() && itRoi.hasNext()) {
			if (itRoi.getNext() != 0) {
				float val = it.getNext();
				if (!(Double.isInfinite(val) || Double.isNaN(val))) {
					sum += val;
					squaredSum += val * val;
				}
			} else {
				it.iterate();
			}
		}
		if (it.hasNext() ^ itRoi.hasNext()) {
			System.err.println("Different Number of elements at computation of SumSquaredSum");
		}
		if (numErrors != 0) {
			System.err.println(
					"Errors found computing SumSquaredSum: " + (100 * numErrors) / clone.getNumberOfElements() + "%");
		}
		return new double[] { sum, squaredSum };
	}

	/**
	 * Gets number of elements in roi; Ignores NaN and infinity
	 */
	public long getNumRoiElements(final NumericGrid roi) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(roi);
		long counter = 0;
		while (it.hasNext()) {
			float val = it.getNext();
			if (val != 0 && Double.isFinite(val) && !Double.isNaN(val)) {
				counter++;
			}
		}
		return counter;
	}

	/** computes mean value and variance in the whole grid */
	public double[] computeMeanVariance(final NumericGrid grid) {
		double[] sums = op.computeSumSquaredSum(grid);
		double sum = sums[0];
		double squaredSum = sums[1];
		double mean = sum / grid.getNumberOfElements();
		double variance = squaredSum / (double) grid.getNumberOfElements() - mean * mean;
		return new double[] { mean, variance };
	}

	/** computes mean value of a grid in a given roi */
	public static double[] computeMeanVariance(final NumericGrid grid, final NumericGrid roi) {
		double[] sums = op.computeSumSquaredSum(grid, roi);
		double sum = sums[0];
		double squaredSum = sums[1];
		double mean = sum / op.getNumRoiElements(roi);
		double variance = squaredSum / (double) op.getNumRoiElements(roi) - mean * mean;
		return new double[] { mean, variance };
	}

	/**
	 * computes mean value of a grid in a given roi; (dimension of roi) <
	 * (dimension of grid)
	 */
	public static double mean(final NumericGrid grid, final NumericGrid roi, int[] indices) {
		double sum = op.computeSumSquaredSum(grid, roi, indices)[0];
		long numRoiValues = op.getNumRoiElements(roi);
		return sum / numRoiValues;
	}

	/** Computes dot product between first and second in given roi */
	public float dotProduct(final NumericGrid first, final NumericGrid second, final NumericGrid roi) {
		double value = 0.0d;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(first);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(second);
		NumericPointwiseIteratorND itRoi = new NumericPointwiseIteratorND(roi);
		while (it1.hasNext() && it2.hasNext() && itRoi.hasNext())
			if (itRoi.getNext() != 0)
				value += it1.getNext() * it2.getNext();
			else {
				it1.iterate();
				it2.iterate();
			}
		return (float) value;
	}

	/**
	 * @param first
	 *            grid to be compared
	 * @param second
	 *            grid to be compared
	 * @param roi
	 *            region of interest in which the root mean square error should
	 *            be computed
	 * @return root mean square error
	 */
	public float rmse(final NumericGrid first, final NumericGrid second, final NumericGrid roi) {
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(first);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(second);
		NumericPointwiseIteratorND itRoi = new NumericPointwiseIteratorND(roi);

		double sum = 0.0f;
		long numErrors = 0;
		long numRoiValues = 0;

		int[] sizeFirst = first.getSize();
		int[] sizeRoi = roi.getSize();
		int difference = sizeFirst.length - sizeRoi.length;

		if (first.getNumberOfElements() != second.getNumberOfElements()) {
			System.err.println("Grids to be compared have different sizes. Result may be falsified!");
		}
		if (difference == 0) {
			if (first.getNumberOfElements() != roi.getNumberOfElements()) {
				System.err.println("Input Grid and roi have different sizes!");
			}
			while (it1.hasNext() && it2.hasNext() && itRoi.hasNext()) {
				if (itRoi.getNext() != 0) {
					numRoiValues++;
					double val = it1.getNext() - it2.getNext();
					if (Double.isInfinite(val) || Double.isNaN(val)) {
						numErrors++;
					} else {
						sum += val * val;
					}
				} else {
					it1.iterate();
					it2.iterate();
				}
			}
		} else {
			System.err.println(
					"Dimension of original image doesn't match dimension of roi. Deliver indices for meaningful results of RMSE. 0 will be returned.");
			return 0;
		}
		if (numErrors != 0) {
			System.err.println("Errors found computing RMSE in roi: " + (numErrors / numRoiValues) * 100 + "%");
		}

		if (numRoiValues == 0) {
			System.err.println(
					"Unsatisfactory roi- no grid value has been considered for computation of RMSE! 0 will be returned.");
			return 0;
		}

		return (float) Math.sqrt(sum / numRoiValues);
	}

	/**
	 * Computes root mean square error of two grids in a roi. Dimension of roi
	 * is smaller than dimension of grids to be compared. Therefore subgrids
	 * will be generated at indices.
	 */
	public float rmse(final NumericGrid original, final NumericGrid reference, final NumericGrid roi, int[] indices) {
		NumericGrid cloneOri = original.clone();
		NumericGrid cloneRef = reference.clone();
		NumericGrid cloneRoi = roi.clone();

		int[] sizeOri = cloneOri.getSize();
		int[] sizeRef = cloneRef.getSize();
		int[] sizeRoi = cloneRoi.getSize();

		int difOR = sizeOri.length - sizeRoi.length;
		int difRR = sizeRef.length - sizeRoi.length;
		if (difOR < 0)
			System.err.println("Dimension of ROI bigger than dimension of original at computation of RMSE.");
		if (difRR < 0)
			System.err.println("Dimension of ROI bigger than dimension of reference at computation of RMSE.");
		if (difOR != indices.length)
			System.err.println(
					"Unusable Array of indices delivered. Doesn't match difference between roi and original respectively reference image");
		if (sizeOri.length == sizeRef.length)
			for (int i = 0; i < difOR; i++) {
				if (indices[i] < 0 || indices[i] > sizeOri[sizeOri.length - 1] - 1) {
					System.err.println("Invalid Index. Index at position " + i + " has to be in intervall [0, "
							+ (sizeOri[sizeOri.length - 1] - 1) + "]");
				}
				cloneOri = cloneOri.getSubGrid(indices[i]);
				cloneRef = cloneRef.getSubGrid(indices[i]);
			}
		return rmse(cloneOri, cloneRef, cloneRoi);
	}

	/**
	 * @param grid
	 *            grid to be analyzed
	 * @return standard deviation
	 */
	public float stddev(final NumericGrid grid) {
		double stdDev = 0.0d;
		long numErrors = 0;
		long numElements = grid.getNumberOfElements();
		double mean = op.computeMeanVariance(grid)[0];
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			double val = it.getNext() - mean;
			if (!(Double.isInfinite(val) || Double.isNaN(val)))
				stdDev += val * val;
			else {
				numErrors++;
			}
		}
		if (numErrors != 0) {
			System.err.println("Errors found computing Standard Deviation: "
					+ (numErrors * 100) / grid.getNumberOfElements() + "%");
		}
		return (float) Math.sqrt(stdDev / numElements);
	}

	/**
	 * This method expects the original grid to have the same dimension as the
	 * roi (e.g. two 3-dimensional grids); otherwise use StdDev(NumericGrid,
	 * NumericGrid, int[]) which first will compute subgrids of roi, depending
	 * on submitted indices array, in order to match dimension
	 * 
	 * @param grid
	 *            grid to be analyzed
	 * @param roi
	 *            region of interest in which the standard deviation should be
	 *            computed
	 * @return standard deviation
	 */
	public float stddev(final NumericGrid grid, final NumericGrid roi) {
		int[] sizeGrid = grid.getSize();
		int[] sizeRoi = roi.getSize();
		int difference = sizeGrid.length - sizeRoi.length;

		double stdDev = 0.0d;
		long numRoiValues = op.getNumRoiElements(roi);
		long numErrors = 0;

		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		NumericPointwiseIteratorND itRoi = new NumericPointwiseIteratorND(roi);

		if (numRoiValues == 0) {
			System.err.println(
					"Unsatisfactory roi - no grid value has been considered for computation of standard deviation!");
		}
		if (difference == 0) {
			double mean = NumericGridOperator.computeMeanVariance(grid, roi)[0];
			if (grid.getNumberOfElements() != roi.getNumberOfElements()) {
				System.err.println("Grids and roi have different dimensions. Result may be falsified!");
			}
			while (it.hasNext() && itRoi.hasNext()) {
				if (itRoi.getNext() != 0) {
					double val = it.getNext() - mean;
					if (!(Double.isInfinite(val) || Double.isNaN(val)))
						stdDev += val * val;
					else {
						numErrors++;
					}
				} else {
					it.iterate();
				}
			}
			if (numErrors != 0) {
				System.err.println("Errors found computing Standard Deviation in ROI: "
						+ (numErrors * 100) / roi.getNumberOfElements() + "%");
			}
			return (float) Math.sqrt(stdDev / numRoiValues);

			// in case of different dimensions a subgrid should be generated
		} else {
			System.err.println(
					"Dimension of original image and dimension of roi don't match! Deliver indices for computation of a subgrid. 0 will be returned!");
			return 0;
		}

	}

	/**
	 * Use this method in case dimension of original image is bigger than
	 * dimension of roi. Subgrid(s) will be created depending on indices.
	 * 
	 * @param indices
	 *            indices should contain (dimension of original image -
	 *            dimension of roi) elements
	 */
	public float stddev(final NumericGrid grid, final NumericGrid roi, int[] indices) {
		int[] sizeG = grid.getSize();
		int[] sizeR = roi.getSize();
		NumericGrid clone = grid.clone();

		int dif = sizeG.length - sizeR.length;
		if (dif < 0)
			System.err.println("Dimension of ROI bigger than dimension of image to be compared.");

		for (int i = 0; i < dif; i++) {
			int[] size = clone.getSize();
			if (indices[i] < 0 || indices[i] > size[size.length - 1] - 1) {
				System.err.println("Invalid Index. Index at position " + i + " has to be in intervall [0, "
						+ (size[size.length - 1] - 1) + "]");
			}
			clone = clone.getSubGrid(indices[i]);
		}

		return stddev(clone, roi);
	}

	/**
	 * returns an array containing the mean and variance of two grids, as well
	 * as the covariance
	 */
	public double[] computeMeanVarianceCovariance(final NumericGrid image, final NumericGrid reference) {
		if (image.getNumberOfElements() != reference.getNumberOfElements())
			System.err.println(
					"Original and reference have different number of elements. Errors in computation of mean, variance and covariance!");
		double[] meanVarianceImage = op.computeMeanVariance(image);
		double meanImage = meanVarianceImage[0];
		double varianceImage = meanVarianceImage[1];
		double[] meanVarianceRef = op.computeMeanVariance(reference);
		double meanRef = meanVarianceRef[0];
		double varianceRef = meanVarianceRef[1];
		double productSum = op.dotProduct(image, reference);
		double covariance = productSum / image.getNumberOfElements() - meanImage * meanRef;
		return new double[] { meanImage, varianceImage, meanRef, varianceRef, covariance };
	}

	/**
	 * returns an array containing the mean and variance of two grids, as well
	 * as the covariance in a given roi
	 * 
	 * @roi region of interest in which mean and variance should be computed
	 */
	public double[] computeMeanVarianceCovariance(final NumericGrid original, final NumericGrid reference,
			final NumericGrid roi) {
		int[] sizeOriginal = original.getSize();
		int[] sizeRoi = roi.getSize();
		int difference = sizeOriginal.length - sizeRoi.length;
		long numRoiValues = op.getNumRoiElements(roi);
		if (numRoiValues == 0)
			System.err.println(
					"Unsatisfactory roi - no grid values have been considered for computation of mean, variance and covariance!");
		if (original.getNumberOfElements() != reference.getNumberOfElements())
			System.err.println(
					"Original and reference have different number of elements. Errors in computation of mean, variance and covariance!");
		if (difference == 0) {
			double[] meanVarianceImage = NumericGridOperator.computeMeanVariance(original, roi);
			double meanImage = meanVarianceImage[0];
			double varianceImage = meanVarianceImage[1];
			double[] meanVarianceRef = NumericGridOperator.computeMeanVariance(reference, roi);
			double meanRef = meanVarianceRef[0];
			double varianceRef = meanVarianceRef[1];
			double productSum = op.dotProduct(original, reference, roi);
			long numRoiElements = op.getNumRoiElements(roi);
			double covariance = productSum / numRoiElements - meanImage * meanRef;
			return new double[] { meanImage, varianceImage, meanRef, varianceRef, covariance };
		} else {
			System.err.println(
					"Computation of mean, variance and covariance failed. Dimensions of original/reference don't match dimension of roi. Submit index in order to generate subgrids. Empty array will be returned");
			return new double[5];
		}
	}

	/**
	 * Use this method in case dimension of original image is bigger than
	 * dimension of roi. Subgrid(s) will be created depending on indices
	 */
	public double[] computeMeanVarianceCovariance(final NumericGrid original, final NumericGrid reference,
			final NumericGrid roi, int[] indices) {
		NumericGrid cloneOri = original.clone();
		NumericGrid cloneRef = reference.clone();
		int[] sizeOriginal = original.getSize();
		int[] sizeRoi = roi.getSize();
		int difference = sizeOriginal.length - sizeRoi.length;
		if (difference != indices.length) {
			System.err.println(
					"Array of indices at computation of MeanVarianceCovariance should have length " + difference);
		}
		// create subgrids as long as dimensions of original/reference are
		// different from dimension of roi
		for (int i = 0; i < difference; i++) {
			if (indices[i] < 0 || indices[i] > sizeOriginal[sizeOriginal.length - 1] - 1) {
				System.err.println("Invalid Index. Index at position " + i + " has to be in intervall [0, "
						+ (sizeOriginal[sizeOriginal.length - 1] - 1) + "]");
			}
			cloneOri = cloneOri.getSubGrid(indices[i]);
			cloneRef = cloneRef.getSubGrid(indices[i]);
		}

		return op.computeMeanVarianceCovariance(cloneOri, cloneRef, roi);

	}

	/**
	 * Implementation of SSIM (structural similarity) based on the paper "Image
	 * Quality Assessment - From Error Visibility to Structural Similarity" by
	 * Wang et. al. - pp. 600-612 alpha, beta and gamma are set 1 --> same
	 * weighting for luminance, contrast and structure; in order to match
	 * SSIM-implementation in class "Arithmetics" submitted bit-depth has to be
	 * 0
	 */

	static final double k1 = 0.01;
	static final double k2 = 0.03;

	/** computes structural similarity of the whole grid */
	public float computeSSIM(final NumericGrid image, final NumericGrid reference) {
		// set bit depth 1 by default --> Constants C1 and C2 are 1
		return op.computeSSIM(image, reference, 1);
	}

	/**
	 * computes SSIM (structural similarity) in the whole grid, while bit-depth
	 * is set manually
	 */
	public float computeSSIM(final NumericGrid image, final NumericGrid reference, int bitDepth) {
		double[] measurements = op.computeMeanVarianceCovariance(image, reference);
		double meanImage = measurements[0];
		double varianceImage = measurements[1];
		double meanRef = measurements[2];
		double varianceRef = measurements[3];
		double covariance = measurements[4];
		// assignment of constant values
		final float alpha = 1f;
		final float beta = 1f;
		final float gamma = 1f;
		final double c1 = Math.pow((Math.pow(2, bitDepth) - 1) * k1, 2);
		final double c2 = Math.pow((Math.pow(2, bitDepth) - 1) * k2, 2);
		final double c3 = c2 / 2;

		double luminance = (2 * meanImage * meanRef + c1) / (meanImage * meanImage + meanRef * meanRef + c1);
		double contrast = (2 * Math.sqrt(varianceImage) * Math.sqrt(varianceRef) + c2)
				/ (varianceImage + varianceRef + c2);
		double structure = (covariance + c3) / (Math.sqrt(varianceImage) * Math.sqrt(varianceRef) + c3);

		return (float) (Math.pow(luminance, alpha) * Math.pow(contrast, beta) * Math.pow(structure, gamma));
	}

	/**
	 * computes SSIM (strucural similarity) in a given roi (region of interest)
	 */
	public float computeSSIM(final NumericGrid image, final NumericGrid reference, final NumericGrid roi) {
		// set bit depth 1 by default --> Constants C1 and C2 are 1
		return op.computeSSIM(image, reference, roi, 1);
	}

	/**
	 * computes SSIM (strucural similarity) in a given roi (region of interest),
	 * while bit-depth is set manually
	 */
	public float computeSSIM(final NumericGrid image, final NumericGrid reference, final NumericGrid roi,
			int bitDepth) {
		double[] measurements = op.computeMeanVarianceCovariance(image, reference, roi);
		double meanImage = measurements[0];
		double varianceImage = measurements[1];
		double meanRef = measurements[2];
		double varianceRef = measurements[3];
		double covariance = measurements[4];
		// assignment of constant values
		final float alpha = 1f;
		final float beta = 1f;
		final float gamma = 1f;
		final double c1 = Math.pow((Math.pow(2, bitDepth) - 1) * k1, 2);
		final double c2 = Math.pow((Math.pow(2, bitDepth) - 1) * k2, 2);
		final double c3 = c2 / 2;

		double luminance = (2 * meanImage * meanRef + c1) / (meanImage * meanImage + meanRef * meanRef + c1);
		double contrast = (2 * Math.sqrt(varianceImage) * Math.sqrt(varianceRef) + c2)
				/ (varianceImage + varianceRef + c2);
		double structure = (covariance + c3) / (Math.sqrt(varianceImage) * Math.sqrt(varianceRef) + c3);

		return (float) (Math.pow(luminance, alpha) * Math.pow(contrast, beta) * Math.pow(structure, gamma));
	}

}

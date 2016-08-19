/*
 * Copyright (C) 2010-2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGridInterface;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGridOperators;


/** The collection of all operators working point-wise on NumericGrid data. */
public abstract class NumericPointwiseOperators {
	/*
	 * Auxiliary method to select a combined grid operator
	 */
	public static NumericGridOperator selectGridOperator(NumericGrid ... grids) {
		boolean nonCLFound = false;
		for (NumericGrid grid : grids){
			if (!(grid instanceof OpenCLGridInterface)){
				nonCLFound = true;
			}
		}
		return (nonCLFound ? NumericGridOperator.getInstance() : OpenCLGridOperators.getInstance());
	}
	
	/** Fill a NumericGrid with the given value */
	public static void fill(final NumericGrid grid, float val) {
		grid.getGridOperator().fill(grid, val);
	}

	/** Get sum of all grid elements */
	public static float sum(final NumericGrid grid) {
		return grid.getGridOperator().sum(grid);
	}
	
	/** Get min of a NumericGrid */
	public static float min(final NumericGrid grid) {
		return grid.getGridOperator().min(grid);
	}

	/** Get max of a NumericGrid */
	public static float max(final NumericGrid grid) {
		return grid.getGridOperator().max(grid);
	}
	
	/** Copy data of a NumericGrid to another, not including boundaries */
	public static void copy(NumericGrid grid1, NumericGrid grid2) {
		NumericGridOperator op = selectGridOperator(grid1, grid2);
		op.copy(grid1, grid2);
	}

	/** Compute dot product between grid1 and grid2 */
	public static float dotProduct(NumericGrid grid1, NumericGrid grid2) {
		NumericGridOperator op = selectGridOperator(grid1, grid2);
		return op.dotProduct(grid1, grid2);
	}
	
	/** Compute dot product between grid1 and grid2 */
	public static float weightedSSD(NumericGrid grid1, NumericGrid grid2, double weightGrid2, double addGrid2) {
		NumericGridOperator op = selectGridOperator(grid1, grid2);
		return op.weightedSSD(grid1, grid2, weightGrid2, addGrid2);
	}
	
	/** Compute dot product between grid1 and grid2 */
	public static float weightedDotProduct(NumericGrid grid1, NumericGrid grid2, double weightGrid2, double subGrid2) {
		NumericGridOperator op = selectGridOperator(grid1, grid2);
		return op.weightedSSD(grid1, grid2, weightGrid2, subGrid2);
	}

	/** Compute dot product between grid and itself. Same as square of l2 norm */
	public static float dotProduct(NumericGrid grid) {
		return dotProduct(grid, grid);
	}

	/** Compute grid3 = grid1 + grid2  */
	public static NumericGrid addedBy(NumericGrid input, NumericGrid sub) {
		NumericGridOperator op = selectGridOperator(input, sub);
		NumericGrid output=input.clone();
		op.addBy(output, sub);
		return output; 
	}
	
	/** Compute grid1 = grid1 + grid2  */
	public static void addBy(NumericGrid input, NumericGrid sub) {
		NumericGridOperator op = selectGridOperator(input, sub);
		op.addBy(input, sub);
	}
	
	
	/** Compute grid = grid + a  */
	public static void addBy(NumericGrid grid, float a) {
		grid.getGridOperator().addBy(grid, a);
	}

	/** Compute grid3 = grid1 - grid2  */
	public static NumericGrid subtractedBy(NumericGrid input, NumericGrid sub) {
		NumericGridOperator op = selectGridOperator(input, sub);
		NumericGrid output = input.clone();
		op.subtractBy(output, sub);
		return output;
	}
	
	/** Compute grid1 = grid1 - grid2  */
	public static void subtractBy(NumericGrid input, NumericGrid sub) {
		NumericGridOperator op = selectGridOperator(input, sub);
		op.subtractBy(input, sub);
	}
	
	
	
	/** Compute grid = grid - a  */
	public static void subtractBy(NumericGrid grid, float a) {
		grid.getGridOperator().subtractBy(grid, a);
	}
	
	public static NumericGrid dividedBy(NumericGrid input, NumericGrid divisor) {
		NumericGridOperator op = selectGridOperator(input, divisor);
		NumericGrid output = input.clone();
		op.divideBy(output, divisor);
		return output;
	}
	
	public static void divideBy(NumericGrid input, NumericGrid divisor) {
		NumericGridOperator op = selectGridOperator(input, divisor);
		op.divideBy(input, divisor);
	}
	
	/** Compute grid = grid / a  */
	public static void divideBy(NumericGrid grid, float a) {
		grid.getGridOperator().divideBy(grid, a);
	}

	public static NumericGrid multipliedBy(NumericGrid input, NumericGrid multiplier) {
		NumericGridOperator op = selectGridOperator(input, multiplier);
		NumericGrid output = input.clone();
		op.multiplyBy(output, multiplier);
		return output;
	}
	
	public static void multiplyBy(NumericGrid input, NumericGrid multiplier) {
		NumericGridOperator op = selectGridOperator(input, multiplier);
		op.multiplyBy(input, multiplier);
	}

	/** Compute grid = grid * a  */
	public static void multiplyBy(NumericGrid grid, float a) {
		grid.getGridOperator().multiplyBy(grid, a);
	}
	
	/** Set all negative values in grid as zero.  */
	public static void removeNegative(NumericGrid grid ) {
		grid.getGridOperator().removeNegative(grid);
	}
	
	public static float mean(NumericGrid data) {
		return sum(data)/data.getNumberOfElements();
	}

	public static float stddev(NumericGrid data, double mean) {
		return data.getGridOperator().stddev(data, mean);
	}

	public static void abs(NumericGrid data) {
		data.getGridOperator().abs(data);
	}
	
	public static void sqr(NumericGrid grid) {
		grid.getGridOperator().pow(grid,2);
	}
	
	public static void pow(NumericGrid grid, double exponent) {
		grid.getGridOperator().pow(grid, exponent);
	}
	
	public static void sqrt(NumericGrid grid) {
		grid.getGridOperator().sqrt(grid);
	}
	
	public static NumericGrid sqrcopy(NumericGrid grid) {
		NumericGrid out = grid.clone();
		sqr(out);
		return out;
	}
	
	public static NumericGrid sqrtcopy(NumericGrid grid) {
		NumericGrid out = grid.clone();
		sqrt(out);
		return out;
	}

	public static void log(NumericGrid data) {
		data.getGridOperator().log(data);
	}

	public static void exp(NumericGrid data) {
		data.getGridOperator().exp(data);
	}
	
	/** set maximum value, all values > max are set to max */
	public static void setMax(NumericGrid data, float max) {
		data.getGridOperator().setMax(data, max);
	}
	
	/** set minimum value, all values < min are set to min */
	public static void setMin(NumericGrid data, float min) {
		data.getGridOperator().setMin(data, min);
	}

}


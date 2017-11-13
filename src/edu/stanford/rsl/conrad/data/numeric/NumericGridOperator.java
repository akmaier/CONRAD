/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;


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
		while (it.hasNext()){
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
		float res = 0;
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

	/** Get number of grid elements with negative values 
	 *  but it doesnt count if negative infinity?!
	 * */
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

	/** Copy data of a NumericGrid to another, not including boundaries. Overwrites grid1 */
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
	public float weightedDotProduct(final NumericGrid grid1,final NumericGrid grid2, float weightGrid2, float addGrid2) {
		float value = 0.0f;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext())
			value += it1.getNext() * (it2.getNext()*weightGrid2 + addGrid2);
		return value;
	}
	
	/** Compute dot product between grid1 and grid2 */
	public float weightedSSD(final NumericGrid grid1,final NumericGrid grid2, double weightGrid2, double addGrid2) {
		float value = 0.0f;
		NumericPointwiseIteratorND it1 = new NumericPointwiseIteratorND(grid1);
		NumericPointwiseIteratorND it2 = new NumericPointwiseIteratorND(grid2);
		while (it1.hasNext()){
			float val = (float)(it1.getNext() - (it2.getNext()*weightGrid2 + addGrid2));
			value += val*val;
		}
		return value;
	}
	
	/** Compute rmse between grid1 and grid2 */
	public float rmse(final NumericGrid grid1,final NumericGrid grid2) {
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
			System.err.println("Errors in RMSE computation: "
					+ ((float) numErrors * 100)
					/ (grid1.getNumberOfElements()) + "%");
		return (float)Math.sqrt(sum/grid1.getNumberOfElements());
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
			if(Double.isInfinite(a) || Double.isNaN(a))
				a = 0;
			it_inout.setNext((float) (a + b));
		}
		input.notifyAfterWrite();
	}

	
	//--> this seems to be wrong
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
	
	/** Compute grid = grid + a. Ignores nans and infinity*/
	public void addBySave(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()){
			float nextValue = it.get();
			if(Double.isInfinite(nextValue) || Double.isNaN(nextValue))
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
			if(Double.isInfinite(a) || Double.isNaN(a))
				a = 0;
			it_inout.setNext((float) (a - b));
		}
		input.notifyAfterWrite();
	}
	
	/** Compute grid = grid - a. Ignores nans and infinity */
	public void subtractBySave(NumericGrid grid, float a) {
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()){
			float nextValue = it.get();
			if(Double.isInfinite(nextValue) || Double.isNaN(nextValue))
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
		while (it.hasNext()){
			float nextValue = it.get();
			if(Double.isInfinite(nextValue) || Double.isNaN(nextValue))
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
			if ((Double.isInfinite(b) || Double.isNaN(b)
					|| Double.isInfinite(a) || Double.isNaN(a)))
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
		if(Double.isInfinite(b) || Double.isNaN(b))
			System.err.println("[multiplyBySave] called with invalid scalar value");
		NumericPointwiseIteratorND it = new NumericPointwiseIteratorND(grid);
		while (it.hasNext()) {
			float a = it.get();
			if (Double.isInfinite(a) || Double.isNaN(a))
				it.setNext(0);
			else
				it.setNext((float)(a * b));
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
		while (it.hasNext()){
			float value =(float)(it.getNext() - mean);
			theStdDev += value*value;
		}
		return (float)Math.sqrt(theStdDev / data.getNumberOfElements());
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
			it.setNext((float) Math.pow(it.get(), (float)exponent));
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
		for(int i=0; i<grid.getSize()[0]; ++i)
			for(int j=0; j<grid.getSize()[1]; ++j)
				gridT.addAtIndex(j, i, grid.getAtIndex(i, j));
		float norm1 = normL1(grid);
		float norm2 = normL1(gridT);
		if(norm1 != norm2)
			System.err.println("Error in transpose");
		return gridT;
	}
	
	/*convert grid2d[] to grid3d */
	public void convert2DArrayTo3D(NumericGrid gridRes,final NumericGrid[] grid) {
		if(grid.length != gridRes.getSize()[2] || grid[0].getSize()[1] != gridRes.getSize()[1] || grid[0].getSize()[0] != gridRes.getSize()[0])
			System.err.println("Sizes dont match");
		else{
			for(int z = 0; z < grid.length;z++)
				for(int y = 0; y < grid.length;y++)
					for(int x = 0; x < grid.length;x++)
						gridRes.setValue(grid[z].getValue(new int[]{x,y}), new int[]{x,y,z});
			gridRes.notifyAfterWrite();
		}
	}

	
	/**
	 * subtract of two grids with given offset
	 * @param gridRes = result grid
	 * @param gridA, gridB = input grids
	 * @param x,y,z Offset = offsetvalue in x,y,and z direction
	 * @param offsetleft = true if left offset, false if right offset
	 */
	public void subtractOffset(NumericGrid gridResult, final NumericGrid gridA, final NumericGrid gridB, int xOffset, int yOffset,int zOffset,boolean offsetleft) {

		if(gridA.getSize()[0] != gridB.getSize()[0] || gridA.getSize()[1] != gridB.getSize()[1] || gridA.getSize()[2] != gridB.getSize()[2])
			System.err.println("Grids have different sizes so they can not be subtracted.");
		
		for (int x = xOffset; x < gridA.getSize()[0]+xOffset; ++x)
			for (int y = yOffset; y < gridA.getSize()[1]+yOffset; ++y)
				for (int z = zOffset; z < gridA.getSize()[2]+zOffset; ++z){
					
					int xIdx = (x >= gridA.getSize()[0] || x < 0) ? Math.min(Math.max(0, x), gridA.getSize()[0]-1) : x;
					int yIdx = (y >= gridA.getSize()[1] || y < 0) ? Math.min(Math.max(0, y), gridA.getSize()[1]-1) : y;
					int zIdx = (z >= gridA.getSize()[2] || z < 0) ? Math.min(Math.max(0, z), gridA.getSize()[2]-1) : z;

					if(offsetleft)
						gridResult.setValue(gridA.getValue(new int[]{xIdx,yIdx,zIdx}) - gridB.getValue(new int[]{x-xOffset,y-yOffset,z-zOffset}), new int[]{x-xOffset,y-yOffset,z-zOffset});
					else
						gridResult.setValue(gridA.getValue(new int[]{x-xOffset,y-yOffset,z-zOffset}) - gridB.getValue(new int[]{xIdx,yIdx,zIdx}), new int[]{x-xOffset,y-yOffset,z-zOffset});
				}
		gridResult.notifyAfterWrite();
	}
	/**
	 * gradient in x-,y- or z-direction
	 * @param gridRes = result grid
	 * @param grid = input grid
	 * @param value = offsetvalue 
	 * @param offsetleft = true if left offset, false if right offset
	 */
	public void gradX(NumericGrid gridRes,final NumericGrid grid,int value, boolean offsetleft) {
		subtractOffset(gridRes,grid,grid,value,0,0,offsetleft);
		gridRes.notifyAfterWrite();
	}

	public void gradY(NumericGrid gridRes,final NumericGrid grid,int value, boolean offsetleft) {
		subtractOffset(gridRes,grid,grid,0,value,0,offsetleft);
		gridRes.notifyAfterWrite();
	}
	
	public void gradZ(NumericGrid gridRes,final NumericGrid grid,int value, boolean offsetleft) {
		subtractOffset(gridRes,grid,grid,0,0,value,offsetleft);
		gridRes.notifyAfterWrite();
	}

	/**
	 * calculates the divergence in x-,y-, or z-direction
	 * @param gridRes = result grid
	 * @param grid = input grid
	 * @param x,y,z Offset = offsetvalue in x,y,and z direction
	 * @param offsetleft = true if left offset, false if right offset
	 */
	public void divergence(NumericGrid gridRes,final NumericGrid grid, int xOffset,int yOffset, int zOffset, boolean offsetleft){

		if(xOffset == 0 && yOffset == 0 && zOffset == 0)
			System.err.println("No offset value chosen");
		else if( (xOffset != 0 && (yOffset != 0 || zOffset != 0)) || (yOffset != 0 && zOffset != 0))
			System.err.println("Too many divergence offsets chosen");
		else{

			//x:0 y:1 z:2
			int mode = 0;
			if(xOffset != 0){
				gridRes.getGridOperator().gradX(gridRes,grid,xOffset,offsetleft);
				mode = 0;
			} else if(yOffset != 0){
				gridRes.getGridOperator().gradY(gridRes,grid,yOffset,offsetleft);
				mode = 1 ;
			} else{
				gridRes.getGridOperator().gradZ(gridRes,grid,zOffset,offsetleft);
				mode = 2;
			}
			int sizeE = gridRes.getSize()[0];
			int sizeF = gridRes.getSize()[1];
	
			for(int e=0; e < sizeE; ++e)
				for(int f=0; f < sizeF; ++f){
					if(mode == 0){
						gridRes.setValue(grid.getValue(new int[]{0,e,f}), new int[]{0,e,f});
						gridRes.setValue(-grid.getValue(new int[]{gridRes.getSize()[0]-2,e,f}), new int[]{gridRes.getSize()[0]-1,e,f});
					} else if(mode == 1) {
						gridRes.setValue(grid.getValue(new int[]{e,0,f}),new int[]{e,0,f});
						gridRes.setValue(-grid.getValue(new int[]{e,gridRes.getSize()[1]-2,f}), new int[]{e,gridRes.getSize()[1]-1,f});
					} else {
						gridRes.setValue(grid.getValue(new int[]{e,f,0}), new int[]{e,f,0});
						gridRes.setValue(-grid.getValue(new int[]{e,f,gridRes.getSize()[2]-2}), new int[]{e,f,gridRes.getSize()[2]-1});
					}
				}
		}
	}
}

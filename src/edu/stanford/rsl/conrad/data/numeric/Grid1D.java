/*
 * Copyright (C) 2010-2014 - Andreas Keil 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * The one-dimensional version of a Grid.  
 * 
 * @see Grid
 * 
 * @author Andreas Keil
 */
public class Grid1D extends NumericGrid {

	/** The actual data buffer.  This is supposed to be initialized outside of and before it is wrapped by Grid1D. */
	protected float[] buffer;

	/** offset for shifted access due to the compatibility with a Grid2D. */
	private int offset = 0;
	
	public Grid1D(int width) {
		buffer = new float[width];
		this.offset = 0;
		this.size = new int[] {buffer.length};
		initialize();
		notifyAfterWrite();
	}
	
	public Grid1D(float[] buffer, int offset, int width) {
		this.buffer = buffer;
		this.offset = offset;
		this.size = new int[] {width};
		initialize();
		notifyAfterWrite();
	}
	
	public Grid1D(float[] buffer) {
		this.buffer = buffer;
		this.offset = 0;
		this.size = new int[] {buffer.length};
		initialize();
		notifyAfterWrite();
	}
	
	public void initialize() {
		this.spacing = new double[1];
		this.origin = new double[1];
		for (int i = 0; i < 1; ++i) {
			assert this.size[i] > 0 : "Size values have to be greater than zero!";
		}
	}

	public  Grid1D(Grid1D input){
		float[] newBuffer = new float[input.getSize()[0]];
		for (int i=0;i<newBuffer.length;++i) 
			newBuffer[i] = input.getAtIndex(i); 
		this.buffer = newBuffer;
		this.size = input.getSize().clone();
		this.spacing = input.spacing.clone();
		this.origin = input.origin.clone();
		this.offset = 0;
	}
	
	/**
	 * EXPLYCITELY copies the elements and returns the copied float array!
	 * Thus, writing on the returned array does not mean writing onto the Grid1D buffer!
	 *	 
	 * @return the buffer
	 */
	public float[] getBuffer() {
		notifyBeforeRead();
		float[] tmp = new float[size[0]];
		System.arraycopy(this.buffer, this.offset, tmp, 0, this.size[0]);
		return tmp;
	}

	
	public double indexToPhysical(double i) {
		return i * this.spacing[0] + this.origin[0];
	}
	public double physicalToIndex(double x) {
		return (x - this.origin[0]) / this.spacing[0];
	}

	public float getAtIndex(int i) {
		notifyBeforeRead();
		return this.buffer[offset + i];
	}
	public void setAtIndex(int i, float val) {
		this.buffer[offset + i] = val;
		notifyAfterWrite();
	}

	public void addAtIndex(int i, float val){
		notifyBeforeRead();
		this.buffer[offset + i] += val;
		notifyAfterWrite();
	}
	public void subAtIndex(int i, float val){
		notifyBeforeRead();
		this.buffer[offset + i] -= val;
		notifyAfterWrite();
	}
	public void multiplyAtIndex(int i, float val){
		notifyBeforeRead();
		this.buffer[offset + i] *= val;
		notifyAfterWrite();
	}
	public void divideAtIndex(int i, float val){
		notifyBeforeRead();
		this.buffer[offset + i] /= val;
		notifyAfterWrite();
	}
	
	@Override
	public String toString() {
		notifyBeforeRead();
		String result = new String();
		result += "[";
		for (int i = 0; i < size[0]; ++i) {
			if (i != 0) result += ", ";
			result += new Float(this.buffer[offset + i]);
		}
		result += "]";
		return result;
	}
	
	/**
	 * EXPLYCITELY copies the elements requested by the subgrid range and returns the copied values in a new Grid1D.
	 * @return the copy of the subgrid
	 */
	public Grid1D getSubGrid(final int startIndex, final int length){
		notifyBeforeRead();
		Grid1D subgrid = new Grid1D(new float [length]);
		for (int i=0; i < length; ++i){
			subgrid.setAtIndex(i, getAtIndex(offset + startIndex + i));
		}
		return subgrid;
	}

	@Override
	public float getValue(int[] idx) {
		notifyBeforeRead();
		return this.getAtIndex(idx[0]);
	}

	@Override
	public NumericGrid clone() {
		notifyBeforeRead();
		return (new Grid1D(this));
	}

	@Override
	public void show(String s) {
		double[] visArray = new double[size[0]];
		for (int i = 0; i < visArray.length; i++) {
			visArray[i]=buffer[offset+i];
		}
		VisualizationUtil.createPlot(s, visArray).show();
	}
	
	@Override
	public void show() {
		show("Grid1D");
	}

	@Override
	public void setValue(float val, int[] idx) {
		setAtIndex(idx[0], val);
	}

	@Override
	public NumericGrid getSubGrid(int index) {
		// not possible
		return null;
	}
	
}



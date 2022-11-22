/*
 * Copyright (C) 2014  Shiyang Hu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.numeric;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * The four-dimensional version of a Grid.
 * 
 * Adopted from Grid3D
 * 
 * @author Shiyang Hu
 */
public class Grid4D extends NumericGrid {

	/**
	 * The actual data buffer. This is supposed to be initialized outside of and
	 * before it is wrapped by Grid1D.
	 */
	protected ArrayList<Grid3D> buffer = null;

	/**
	 * Constructor that directly allocates the Memory
	 * @param width
	 * @param height
	 * @param depth
	 * @param dimension 
	 */
	public Grid4D(int width, int height, int depth, int dimension){
		this(width, height, depth, dimension, true);
	}

	/**
	 * Constructor does not allocate the Memory but saves the size
	 * @param width
	 * @param height
	 * @param depth
	 * @param allocateImmediately Boolean flag for memory allocation
	 */
	public Grid4D(int width, int height, int depth, int dimension, boolean allocateImmediately)
	{
		buffer = new ArrayList<Grid3D>(dimension);

		this.size = new int[] { width, height, depth, dimension};
		this.spacing = new double[4];
		this.origin = new double[4];
		for (int i = 0; i < 4; ++i) {
			assert this.size[i] > 0 : "Size values have to be greater than zero!";
		}

		if (allocateImmediately) {
			allocate();
		} else {
			for (int i=0; i < size[3]; ++i) {
				buffer.add(null);
			}		
		}
	}

	/**
	 * Allocate the memory for all slices
	 */
	public void allocate ()
	{
		for (int i=0; i < size[3]; ++i) {
			buffer.add(new Grid3D(size[0],size[1],size[2]));
		}
	}


	public ArrayList<Grid3D> getBuffer() {
		notifyBeforeRead();
		return this.buffer;
	}


	public Grid4D(Grid4D input){
		this.size = input.size;
		this.spacing = input.spacing;
		this.origin = input.origin;

		this.buffer = new ArrayList<Grid3D>(size[3]);
		for (int i=0; i < size[3]; ++i) {
			buffer.add(new Grid3D(input.getSubGrid(i)));
		}
	}

	public Grid3D getSubGrid(int i) {
		notifyBeforeRead();
		return this.buffer.get(i);
	}

	public void setSubGrid(int i, Grid3D grid) {
		this.buffer.set(i, grid);
		notifyAfterWrite();
	}

	public double[] indexToPhysical(double i, double j, double k, double l) {
		return new double[] { i * this.spacing[0] + this.origin[0],
				j * this.spacing[1] + this.origin[1],
				k * this.spacing[2] + this.origin[2],
				l * this.spacing[3] + this.origin[3],
		};
	}

	public double[] physicalToIndex(double x, double y, double z, double w) {
		return new double[] { (x - this.origin[0]) / this.spacing[0],
				(y - this.origin[1]) / this.spacing[1],
				(z - this.origin[2]) / this.spacing[2],
				(w - this.origin[3]) / this.spacing[3]};
	}

	public float getAtIndex(int i, int j, int k, int m) {
		notifyBeforeRead();
		return this.buffer.get(m).getAtIndex(i, j, k);
	}

	public void setAtIndex(int i, int j, int k, int m, float val) {
		this.buffer.get(m).setAtIndex(i, j,k, val);
		notifyAfterWrite();
	}

	public void addAtIndex(int i, int j, int k, int m, float val) {
		notifyBeforeRead();
		setAtIndex(i, j, k,m,  getAtIndex(i, j, k, m) + val);
		notifyAfterWrite();
	}

	public void multiplyAtIndex(int i, int j, int k, int m, float val) {
		notifyBeforeRead();
		setAtIndex(i, j, k,m,  getAtIndex(i, j, k, m) + val);
		notifyAfterWrite();
	}

	@Override
	public String toString() {
		String result = new String();
		result += "[";
		for (int i = 0; i < this.buffer.size(); ++i) {
			if (i != 0)
				result += ", ";
			result += this.getSubGrid(i).toString();
		}
		result += "]";
		return result;
	}


	public void show(String title){
		ImageUtil.wrapGrid4D(this, title).show();
	}
	
	public void show(){
		show("Grid3D");
	}

	@Override
	public float getValue(int[] idx) {
		notifyBeforeRead();
		return this.getAtIndex(idx[0], idx[1], idx[2], idx[3]);
	}

	@Override
	public NumericGrid clone() {
		notifyBeforeRead();
		return (new Grid4D(this));
	}

	@Override
	public void setSpacing(double... spacing){
		super.setSpacing(spacing);
		for (Grid3D slice : buffer){
			if (slice != null)
				slice.setSpacing(spacing[0],spacing[1],spacing[2]);
		}
	}

	@Override
	public void setOrigin(double... origin){
		super.setOrigin(origin);
		for (Grid3D slice : buffer){
			if (slice != null)
				slice.setOrigin(origin[0],origin[1],origin[2]);
		}
	}

	@Override
	public void setValue(float val, int[] idx) {
		setAtIndex(idx[0], idx[1], idx[2], idx[3], val);
	}

}



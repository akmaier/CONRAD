/*
 * Copyright (C) 2010-2014 - Andreas Keil 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Transformable;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * The two-dimensional version of a Grid.  
 * 
 * @see Grid
 * 
 * @author Andreas Keil
 */
public class Grid2D extends NumericGrid implements Transformable {

	/** The actual data buffer.  This is supposed to be initialized outside of the class. */
	protected float[] buffer;
	
	/** The offsets in the linear memory for each column. */
	protected int[] columnOffsets;
	
	/** Link to the underlying columns represented as 1D grids */
	protected Grid1D[] subGrids;
	
	public Grid2D(int width, int height) {
		assert width > 0 && height > 0;
		this.buffer = new float[width*height];
		this.initialize(width, height);
		notifyAfterWrite();
	}
	
	public Grid2D(float[] buffer, int[] size) { 
		this(buffer, size[0], size[1]);
		notifyAfterWrite();
	}
	
	public Grid2D(float[] buffer, int width, int height) {
		assert buffer.length == width*height;
		this.buffer = buffer;
		this.initialize(width, height);
		notifyAfterWrite();
	}
	
	
	protected void initialize(int width, int height)
	{
		this.size = new int[] {width, height};
		this.spacing = new double[2];
		this.origin = new double[2];
		for (int i = 0; i < 2; ++i) {
			assert size[i] > 0 : "Size values have to be greater than zero!";
		}
		columnOffsets = new int[this.size[1]];
		subGrids = new Grid1D[size[1]];
		for (int i = 0; i < columnOffsets.length; ++i) {
			columnOffsets[i] = i*size[0];
			subGrids[i] = new Grid1D(this.buffer, columnOffsets[i], size[0]);
		}
	}
	
	
	public Grid2D(Grid2D input){
		assert input.getWidth()*input.getHeight() == this.buffer.length;
		this.size = input.size.clone();
		this.spacing = input.spacing.clone();
		this.origin = input.origin.clone();
		this.buffer = input.getBuffer().clone();
		this.columnOffsets = new int[this.size[1]];
		this.subGrids = new Grid1D[size[1]];
		for (int i = 0; i < columnOffsets.length; ++i) {
			this.columnOffsets[i] = i*this.size[0];
			this.subGrids[i] = new Grid1D(this.buffer, columnOffsets[i], size[0]);
		}
	}
	
	/**
	 * Returns a reference to the linear buffer containing the 2D image in a row-first manner.
	 * @return the buffer as float array
	 */
	public float[] getBuffer() {
		notifyBeforeRead();
		return this.buffer;
	}

	/**
	 * Returns the corresponding Grid1D object that points on the linear 2D row memory 
	 * @param j The row-index (y-index, height-index) 
	 * @return the sub grid
	 */
	public Grid1D getSubGrid(int j) {
		notifyBeforeRead();
		return subGrids[j];
	}

	
	public double[] indexToPhysical(double i, double j) {
		return new double[] {
				i * this.spacing[0] + this.origin[0],
				j * this.spacing[1] + this.origin[1]
		};
	}
	
	
	public double[] physicalToIndex(double x, double y) {
		return new double[] {
				(x - this.origin[0]) / this.spacing[0],
				(y - this.origin[1]) / this.spacing[1]
		};
	}

	
	public float getAtIndex(int i, int j) {
		notifyBeforeRead();
		//FIXME (maybe use getPixelValue instead)
		return this.getPixelValue(i, j);
	}
	
	
	public void setAtIndex(int i, int j, float val) {
		//FIXME (maybe use putPixelValue instead)
		 this.putPixelValue(i, j, val);
		 notifyAfterWrite();
	}

	
	public void addAtIndex(int i, int j, float val){
		notifyBeforeRead();
		this.buffer[j*this.size[0]+i] += val;
		notifyAfterWrite();
	}
	
	
	public void multiplyAtIndex(int i, int j, float val){
		notifyBeforeRead();
		this.buffer[j*this.size[0]+i] *= val;
		notifyAfterWrite();
	}
	
	
	public void subAtIndex(int i, int j, float val){
		notifyBeforeRead();
		this.addAtIndex(i,j,val*-1.f);
		notifyAfterWrite();
	}
	
	
	public void divideAtIndex(int i, int j, float val){
		notifyBeforeRead();
		this.multiplyAtIndex(i, j, 1.f/val);
		notifyAfterWrite();
	}
	
	
	public String toStringMatlab() {
		notifyBeforeRead();
		String result = new String();
		result += "[";
		for (int i = 0; i < this.buffer.length; ++i) {
			if (i != 0) result += ", ";
			result += this.buffer[i];
		}
		result += "]";
		return result;
	}

	@Override
	public String toString(){
		String result = "Grid 2D " + size[0] + "x" + size[1];
		/*if(debug){
			float min = this.getGridOperator().min(this);
			float max = this.getGridOperator().max(this);
			result += " " + min + ":" + max;
		}*/
		return result;
	}
	
	
	public void show(String title){
		notifyBeforeRead();
		VisualizationUtil.showGrid2D(this, title);
	}
	

	public void show(){
		notifyBeforeRead();
		show("Grid2D");
	}

	
	public float [] getPixels(){
		return null;
	}
	
	
	/**
	 * Gets the grid's width
	 * @return the width
	 */
	public int getWidth()
	{
		return this.size[0];
	}
	
	
	/**
	 * Gets the grid's height
	 * @return the height
	 */
	public int getHeight()
	{
		return this.size[1];
	}
	
	
	/**
	 * Set a pixel value at position (x,y)
	 * @param x The value's x position
	 * @param y The value's y position
	 * @param value The value to set
	 */
	public void putPixelValue(int x, int y, float value) {
		if (y>=size[1] || x >= size[0]) return; 
		this.buffer[y*this.size[0]+x]=value;
		notifyAfterWrite();
	}
	
	
	/**
	 * Set a pixel value at position (x,y)
	 * @param x The value's x position
	 * @param y The value's y position
	 * @param value The value to set
	 */
	public void putPixelValue(int x, int y, double value) {
		this.putPixelValue(x, y, (float)value);
	}
	
	
	/**
	 * Get the pixel value at position (x,y)
	 * This method uses linear indices to access the buffer and does not perform range checking!
	 * @param x The value's x position
	 * @param y The value's y position
	 * @return the value of the pixel
	 */
	public float getPixelValue(int x, int y) {
		notifyBeforeRead();
		//System.out.println("getPixelValue - Min: " + this.getGridOperator().min(this) + " Max: " + this.getGridOperator().max(this)); // TODO test
		return this.buffer[y*this.size[0]+x];
	}
	
	public float getPixelValue(int count) {
		int x = count % this.size[0]; 
		int y = count / this.size[0];
		return getPixelValue(x, y);
	}

	@Override
	public float getValue(int[] idx) {
		return this.getAtIndex(idx[0], idx[1]);
	}

	@Override
	public NumericGrid clone() {
		notifyBeforeRead();
		return (new Grid2D(this));
	}

	@Override
	public void applyTransform(Transform t) {
		notifyBeforeRead();
		Grid2D tmp = new Grid2D(this);
		for (int i=0; i < getSize()[0]; ++i){
			for (int j=0; j < getSize()[1]; ++j){
				PointND newPos = new PointND(indexToPhysical(i, j));
				newPos.applyTransform(t);
				double[] pos = physicalToIndex(newPos.get(0),newPos.get(1));
				float val = InterpolationOperators.interpolateLinear(tmp, pos[0], pos[1]);
				this.setAtIndex(i, j, val);
			}
		}
		notifyAfterWrite();
	}
	
	@Override
	public void setSpacing(double... spacing){
		super.setSpacing(spacing);
		for (Grid1D rows : subGrids)
			rows.setSpacing(spacing[0]);
	}
	
	@Override
	public void setOrigin(double... origin){
		super.setOrigin(origin);
		for (Grid1D rows : subGrids)
			rows.setOrigin(origin[0]);
	}

	@Override
	public void setValue(float val, int[] idx) {
		setAtIndex(idx[0], idx[1], val);
	}

}



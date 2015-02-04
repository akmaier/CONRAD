/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;

/**
 * Class to use complex numbers in a grid structure.
 * Internally float arrays are used. The real and imaginary parts are stored in an alternating manner.
 * @author akmaier / berger
 *
 */
public class ComplexGrid1D extends ComplexGrid {
	
	/** Linear float array stores the complex numbers in an alternating manner [Re1, Imag1, Re2, Imag2, ..., ReN,ImagN]. */
	private float[] buffer;
	
	/** offset for shifted access due to the compatibility with a Grid2DComplex. */
	private int offset = 0;
	
	public ComplexGrid1D(int width) {
		buffer = new float[width*2];
		this.size = new int[] {width};
		this.offset = 0;
		init();
		notifyAfterWrite();
	}
	
	public ComplexGrid1D(float[] buffer, int offset, int width){
		this.buffer = buffer;
		this.size = new int[] {width};
		this.offset = offset;
		init();
		notifyAfterWrite();
	}
	
	public ComplexGrid1D(Grid1D grid){
		this(grid.getSize()[0]);
		final int inputSize = grid.getSize()[0];
		for (int i=0; i < inputSize;++i){
			this.setAtIndex(i, grid.getAtIndex(i));
		}
		this.size = new int[]{grid.getSize()[0]};
		this.origin = new double[]{grid.getOrigin()[0]};
		this.spacing = new double[]{grid.getSpacing()[0]};
		this.offset = 0;
		notifyAfterWrite();
	}

	public ComplexGrid1D(ComplexGrid1D grid){
		this(grid.getSize()[0]);
		final int inputSize = grid.getSize()[0];
		for (int i=0; i < inputSize;++i){
			this.setAtIndex(i, grid.getAtIndex(i));
		}
		this.size = new int[]{grid.size[0]};
		this.origin = new double[]{grid.origin[0]};
		this.spacing = new double[]{grid.spacing[0]};
		this.offset = 0;
		if(grid.openCLactive)
			activateCL();
	}
	
	public void init(){
		this.origin = new double[]{0};
		this.spacing = new double[]{1};
	}
	
	public void multiplyAtIndex(int index, float val) {
		multiplyAtIndex(index, val,0);
	}
	
	public void multiplyAtIndex(int index, float real, float imag) {
		Complex in = new Complex(real,imag);
		multiplyAtIndex(index,in);
	}
	
	public void multiplyAtIndex(int index, Complex in) {
		setAtIndex(index, getAtIndex(index).mul(in));
	}
	
	
	public void divideAtIndex(int index, float val) {
		divideAtIndex(index, val, 0);
	}
	
	public void divideAtIndex(int index, float real, float imag) {
		Complex in = new Complex(real,imag);
		divideAtIndex(index,in);
	}
	
	public void divideAtIndex(int index, Complex in) {
		setAtIndex(index, getAtIndex(index).div(in));
	}
	
	
	public void addAtIndex(int index, float val) {
		addAtIndex(index, val, 0);
	}
	
	public void addAtIndex(int index, float real, float imag) {
		Complex in = new Complex(real,imag);
		addAtIndex(index,in);
	}
	
	public void addAtIndex(int index, Complex in) {
		setAtIndex(index, getAtIndex(index).add(in));
	}
	
	
	public void subtractAtIndex(int index, float val) {
		subtractAtIndex(index, val, 0);
	}
	
	public void subtractAtIndex(int index, float real, float imag) {
		Complex in = new Complex(real,imag);
		subtractAtIndex(index,in);
	}
	
	public void subtractAtIndex(int index, Complex in) {
		setAtIndex(index, getAtIndex(index).sub(in));
	}
	
	
	
	public void setAtIndex(int index, float val){
		setAtIndex(index,val,0);
	}
	
	public void setAtIndex(int index, float real, float imag){
		Complex in = new Complex(real,imag);
		setAtIndex(index, in);
	}
	
	public void setAtIndex(int index, Complex val){
		buffer[(index+offset)*2]=(float)val.getReal();
		buffer[(index+offset)*2+1]=(float)val.getImag();
		notifyAfterWrite();
	}
	
	public Complex getAtIndex(int index){
		notifyBeforeRead();
		return new Complex(buffer[(index+offset)*2],buffer[(index+offset)*2+1]);
	}
	
	
	
	public Grid1D getRealGrid(){
		return getRealSubGrid(0, this.size[0]);
	}
	
	public Grid1D getImagGrid(){
		return getImagSubGrid(0, this.size[0]);
	}
	
	public Grid1D getMagGrid(){
		return getMagSubGrid(0, this.size[0]);
	}
	
	public Grid1D getPhaseGrid(){
		return getPhaseSubGrid(0, this.size[0]);
	}
	
	public Grid1D getRealSubGrid(final int startIndex, final int length){
		Grid1D subgrid = new Grid1D(length);
		for (int i=0; i < length; ++i){
			subgrid.setAtIndex(i, (float)this.getAtIndex(startIndex+i).getReal());
		}
		return subgrid;
	}
	
	public Grid1D getImagSubGrid(final int startIndex, final int length){
		Grid1D subgrid = new Grid1D(new float [length]);
		for (int i=0; i < length; ++i){
			subgrid.setAtIndex(i, (float)this.getAtIndex(startIndex+i).getImag());
		}
		return subgrid;
	}
	
	public Grid1D getMagSubGrid(final int startIndex, final int length){
		Grid1D subgrid = new Grid1D(new float [length]);
		for (int i=0; i < length; ++i){
			subgrid.setAtIndex(i, (float)this.getAtIndex(startIndex+i).getMagn());
		}
		return subgrid;
	}
	
	public Grid1D getPhaseSubGrid(final int startIndex, final int length){
		Grid1D subgrid = new Grid1D(new float [length]);
		for (int i=0; i < length; ++i){
			subgrid.setAtIndex(i, (float)this.getAtIndex(startIndex+i).getAngle());
		}
		return subgrid;
	}

	@Override
	public Complex getValue(int... idx) {
		return this.getAtIndex(idx[0]);
	}
	
	@Override
	public void setValue(Complex val, int... idx) {
		setAtIndex(idx[0], val);
	}

	@Override
	public ComplexGrid clone() {
		notifyBeforeRead();
		return new ComplexGrid1D(this);
	}

	@Override
	public String toString() {
		String result = new String();
		result += "[";
		for (int i = 0; i < size[0]; ++i) {
			if (i != 0) result += ", ";
			result += getAtIndex(i);
		}
		result += "]";
		return result;
	}
	
	public float[] getBuffer() {
		notifyBeforeRead();
		return buffer;
	}
	
	public int getOffset() {
		return offset;
	}

	@Override
	public float[] getAslinearMemory() {
		notifyBeforeRead();
		return buffer;
	}

	@Override
	public void setAslinearMemory(float[] buffer) {
		this.buffer = buffer;
		notifyAfterWrite();
	}
	
	@Override
	public float getRealAtIndex(int... idx) {
		notifyBeforeRead();
		return buffer[2*(offset+idx[0])];
	}

	@Override
	public void setRealAtIndex(float val, int... idx) {
		buffer[2*(offset+idx[0])] = val;
		notifyAfterWrite();
	}

	@Override
	public float getImagAtIndex(int... idx) {
		notifyBeforeRead();
		return buffer[2*(idx[0]+offset)+1];
	}

	@Override
	public void setImagAtIndex(float val, int... idx) {
		buffer[2*(idx[0]+offset)+1] = val;
		notifyAfterWrite();
	}
	
}


/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;



public class ComplexGrid2D extends ComplexGrid {

	/** Linear float array stores the complex numbers in an alternating manner [Re1, Imag1, Re2, Imag2, ..., ReN,ImagN]. */
	private float[] buffer;
	
	private int offset;

	private ComplexGrid1D[] subGrids = null;
	
	
	public ComplexGrid2D(int width, int height) {
		this(new float[width*height*2],0,width,height);
	}
	
	public ComplexGrid2D(float[] buffer, int offset, int width, int height){
		this.buffer = buffer;
		this.offset = offset;
		this.spacing = new double[] {1,1};
		this.origin = new double[2];
		this.size = new int[]{width,height}; 
		initialize(width, height);
		notifyAfterWrite();
	}
	
	public ComplexGrid2D(ComplexGrid2D input){
		this.offset = 0;
		this.size = new int[] {input.getSize()[0],input.getSize()[1]}; 
		this.spacing = new double[] {input.spacing[0],input.spacing[1]};
		this.origin = new double[] {input.origin[0],input.origin[1]};
		buffer = new float[input.getNumberOfElements()*2];
		System.arraycopy(input.getBuffer(), input.getOffset()*2, this.buffer, 0, this.buffer.length);
		initialize(input.getSize()[0],input.getSize()[1]);
		if(input.openCLactive)
			activateCL();
	}
	
	public ComplexGrid2D(Grid2D input){
		this.offset = 0;
		this.size = new int[] {input.getSize()[0],input.getSize()[1]}; 
		this.spacing = new double[] {input.getSpacing()[0],input.getSpacing()[1]};
		this.origin = new double[] {input.getOrigin()[0],input.getOrigin()[1]};
		buffer = new float[input.getNumberOfElements()*2];
		for (int i = 0; i < input.getBuffer().length; i++) {
			this.buffer[i*2]=input.getBuffer()[i];
		}
		initialize(input.getSize()[0],input.getSize()[1]);
	}
	
	protected void initialize(int width, int height)
	{
		for (int i = 0; i < 2; ++i) {
			assert size[i] > 0 : "Size values have to be greater than zero!";
		}
		subGrids = new ComplexGrid1D[size[1]];
		for (int i = 0; i < height; ++i) {
			subGrids[i] = new ComplexGrid1D(this.buffer, i*size[0]+offset, size[0]);
		}
	}
	
	
	public float[] getBuffer(){
		notifyBeforeRead();
		return buffer;
	}
	
	
	
	
	public void multiplyAtIndex(int i, int j, float val) {
		multiplyAtIndex(i, j, val, 0);
	}
	
	public void multiplyAtIndex(int i, int j, float real, float imag) {
		Complex in = new Complex(real,imag);
		multiplyAtIndex(i,j,in);
	}
	
	public void multiplyAtIndex(int i, int j, Complex in) {
		setAtIndex(i, j, getAtIndex(i,j).mul(in));
	}
	
	
	public void divideAtIndex(int i, int j, float val) {
		divideAtIndex(i, j, val, 0);
	}
	
	public void divideAtIndex(int i, int j, float real, float imag) {
		Complex in = new Complex(real,imag);
		divideAtIndex(i, j, in);
	}
	
	public void divideAtIndex(int i, int j, Complex in) {
		setAtIndex(i, j, getAtIndex(i, j).div(in));
	}
	
	
	public void addAtIndex(int i, int j, float val) {
		addAtIndex(i, j, val, 0);
	}
	
	public void addAtIndex(int i, int j, float real, float imag) {
		Complex in = new Complex(real,imag);
		addAtIndex(i, j, in);
	}
	
	public void addAtIndex(int i, int j, Complex in) {
		setAtIndex(i, j, getAtIndex(i, j).add(in));
	}
	
	
	public void subtractAtIndex(int i, int j, float val) {
		subtractAtIndex(i, j, val, 0);
	}
	
	public void subtractAtIndex(int i, int j, float real, float imag) {
		Complex in = new Complex(real,imag);
		subtractAtIndex(i, j,in);
	}
	
	public void subtractAtIndex(int i, int j, Complex in) {
		setAtIndex(i, j, getAtIndex(i, j).sub(in));
	}
	
	
	
	public void setAtIndex(int i, int j, float val){
		setAtIndex(i, j,val,0);
	}
	
	public void setAtIndex(int i, int j, float real, float imag){
		Complex in = new Complex(real,imag);
		setAtIndex(i, j, in);
	}
	
	public void setAtIndex(int i, int j, Complex val){
		getSubGrid(j).setAtIndex(i,val);
		notifyAfterWrite();
	}
	
	public Complex getAtIndex(int i, int j){
		notifyBeforeRead();
		return getSubGrid(j).getAtIndex(i);
	}
	
	public ComplexGrid1D getSubGrid(int j){
		notifyBeforeRead();
		return this.subGrids[j];
	}
	
	
	@Override
	public Complex getValue(int... idx) {
		return getAtIndex(idx[0],idx[1]);
	}


	@Override
	public ComplexGrid clone() {
		notifyBeforeRead();
		return new ComplexGrid2D(this);
	}

	@Override
	public String toString() {
		String result = new String();
		result += "[";
		for (int j = 0; j < size[1]; ++j) {
			if (j != 0) result += "; ";
			result += getSubGrid(j).toString();
		}
		result += "]";
		return result;
	}

	@Override
	public Grid2D getRealGrid() {
		return getRealSubGrid(0, 0, size[0], size[1]);
	}

	@Override
	public Grid2D getImagGrid() {
		return getImagSubGrid(0, 0, size[0], size[1]);
	}

	@Override
	public Grid2D getMagGrid() {
		return getMagSubGrid(0, 0, size[0], size[1]);
	}

	@Override
	public Grid2D getPhaseGrid() {
		return getPhaseSubGrid(0, 0, size[0], size[1]);
	}
	
	public Grid2D getRealSubGrid(final int startX, final int startY, final int lengthX, final int lengthY){
		Grid2D subgrid = new Grid2D(lengthX,lengthY);
		for (int i=0; i < lengthX; ++i){
			for (int j=0; j < lengthY; ++j){
			subgrid.setAtIndex(i,j, (float)this.getAtIndex(startX+i,startY+j).getReal());
			}
		}
		return subgrid;
	}
	
	public Grid2D getImagSubGrid(final int startX, final int startY, final int lengthX, final int lengthY){
		Grid2D subgrid = new Grid2D(lengthX,lengthY);
		for (int i=0; i < lengthX; ++i){
			for (int j=0; j < lengthY; ++j){
			subgrid.setAtIndex(i,j, (float)this.getAtIndex(startX+i,startY+j).getImag());
			}
		}
		return subgrid;
	}
	
	public Grid2D getMagSubGrid(final int startX, final int startY, final int lengthX, final int lengthY){
		Grid2D subgrid = new Grid2D(lengthX,lengthY);
		for (int i=0; i < lengthX; ++i){
			for (int j=0; j < lengthY; ++j){
			subgrid.setAtIndex(i,j, (float)this.getAtIndex(startX+i,startY+j).getMagn());
			}
		}
		return subgrid;
	}
	
	public Grid2D getPhaseSubGrid(final int startX, final int startY, final int lengthX, final int lengthY){
		Grid2D subgrid = new Grid2D(lengthX,lengthY);
		for (int i=0; i < lengthX; ++i){
			for (int j=0; j < lengthY; ++j){
			subgrid.setAtIndex(i,j, (float)this.getAtIndex(startX+i,startY+j).getAngle());
			}
		}
		return subgrid;
	}
	
	@Override
	public void setValue(Complex val, int... idx) {
		setAtIndex(idx[0], idx[1], val);
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
		return getSubGrid(idx[1]).getRealAtIndex(idx[0]);
	}

	@Override
	public void setRealAtIndex(float val, int... idx) {
		getSubGrid(idx[1]).setRealAtIndex(val,idx[0]);
		notifyAfterWrite();
	}

	@Override
	public float getImagAtIndex(int... idx) {
		return getSubGrid(idx[1]).getImagAtIndex(idx[0]);
	}

	@Override
	public void setImagAtIndex(float val, int... idx) {
		getSubGrid(idx[1]).setImagAtIndex(val,idx[0]);
		notifyAfterWrite();
	}
	
	public int getOffset() {
		return offset;
	}
	
	public static void main(String[] args) {
		new ImageJ();
		ComplexGrid2D bla = new ComplexGrid2D(new SheppLogan(64));
		ComplexGrid2D gc = new ComplexGrid2D(bla);
		ComplexGrid2D gc1 = new ComplexGrid2D(gc);
		bla.show();
		Fourier ft = new Fourier();
		ft.fft(gc);
		ft.ifft(gc);
		ft.fft2(gc1);
		ft.ifft2(gc1);
		
		ComplexGridOperator gop = new ComplexGridOperator();
		ComplexGrid diff = bla.clone();
		gop.subtractBy(diff, gc);
		
		ComplexGrid diff1 = bla.clone();
		gop.subtractBy(diff1, gc1);
		
		diff.show();
		diff1.show();
		gc.show();
		gc1.show();

	}
}

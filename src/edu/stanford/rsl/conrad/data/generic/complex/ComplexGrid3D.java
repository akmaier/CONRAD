/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;



public class ComplexGrid3D extends ComplexGrid {

	/** Linear float array stores the complex numbers in an alternating manner [Re1, Imag1, Re2, Imag2, ..., ReN,ImagN]. */
	private float[] buffer;

	private ComplexGrid2D[] subGrids = null;

	public ComplexGrid3D(int width, int height, int depth) {
		assert width > 0 && height > 0 && depth > 0;
		this.size = new int[] {width, height, depth};
		this.spacing = new double[]{1,1,1};
		this.origin = new double[3];
		buffer = new float[width*height*depth*2];
		initialize(width, height, depth);
		notifyAfterWrite();
	}

	public ComplexGrid3D(ComplexGrid3D input){
		this.size = new int[] {input.size[0], input.size[1], input.size[2]};
		this.spacing = new double[] {input.spacing[0],input.spacing[1], input.spacing[2]};
		this.origin = new double[] {input.origin[0],input.origin[1], input.origin[2]};
		buffer = new float[input.getNumberOfElements()*2];
		System.arraycopy(input.getBuffer(), input.getOffset()*2, buffer, 0, buffer.length);
		initialize(input.size[0], input.size[1], input.size[2]);
		if(input.openCLactive)
			activateCL();
	}

	public ComplexGrid3D(Grid3D input){
		this.spacing = new double[] {input.getSpacing()[0],input.getSpacing()[1], input.getSpacing()[2]};
		this.origin = new double[] {input.getOrigin()[0],input.getOrigin()[1], input.getOrigin()[2]};
		this.size = new int[] {input.getSize()[0], input.getSize()[1], input.getSize()[2]};
		buffer = new float[input.getNumberOfElements()*2];
		initialize(input.getSize()[0], input.getSize()[1], input.getSize()[2]);
		for (int k = 0; k < input.getSize()[2]; k++) {
			for (int j = 0; j < input.getSize()[1]; j++) {
				for (int i = 0; i < input.getSize()[0]; i++) {
					this.setAtIndex(i, j, k, input.getAtIndex(i, j, k));
				}
			}
		}
		
	}

	public ComplexGrid3D(Grid3D realInput, Grid3D imagInput){
		this.spacing = new double[] {realInput.getSpacing()[0],realInput.getSpacing()[1], realInput.getSpacing()[2]};
		this.origin = new double[] {realInput.getOrigin()[0],realInput.getOrigin()[1], realInput.getOrigin()[2]};
		this.size = new int[] {realInput.getSize()[0], realInput.getSize()[1], realInput.getSize()[2]};
		buffer = new float[realInput.getNumberOfElements()*2];
		initialize(realInput.getSize()[0], realInput.getSize()[1], realInput.getSize()[2]);
		for(int k = 0; k < realInput.getSize()[2]; k++){
			for(int j = 0; j < realInput.getSize()[1]; j++){
				for(int i = 0; i < realInput.getSize()[0]; i++ ){
					this.setAtIndex(i, j, k, realInput.getAtIndex(i, j, k), imagInput.getAtIndex(i, j, k));
				}
			}
		}
		
	}
	protected void initialize(int width, int height, int depth)
	{
		subGrids = new ComplexGrid2D[depth];
		for (int i = 0; i < subGrids.length; i++) {
			subGrids[i] = new ComplexGrid2D(buffer, i*width*height, width, height);
		}
	}


	public float[] getBuffer(){
		notifyBeforeRead();
		return buffer;
	}


	public void multiplyAtIndex(int i, int j, int k, float val) {
		multiplyAtIndex(i, j, k, val, 0);
	}

	public void multiplyAtIndex(int i, int j, int k, float real, float imag) {
		Complex in = new Complex(real, imag);
		multiplyAtIndex(i, j, k, in);
	}

	public void multiplyAtIndex(int i, int j, int k, Complex in) {
		setAtIndex(i, j, k, getAtIndex(i, j, k).mul(in));
	}


	public void divideAtIndex(int i, int j, int k, float val) {
		divideAtIndex(i, j, k, val, 0);
	}

	public void divideAtIndex(int i, int j, int k, float real, float imag) {
		Complex in = new Complex(real, imag);
		divideAtIndex(i, j, k, in);
	}

	public void divideAtIndex(int i, int j, int k, Complex in) {
		setAtIndex(i, j, k, getAtIndex(i, j, k).div(in));
	}


	public void addAtIndex(int i, int j, int k, float val) {
		addAtIndex(i, j, k, val, 0);
	}

	public void addAtIndex(int i, int j, int k, float real, float imag) {
		Complex in = new Complex(real, imag);
		addAtIndex(i, j, k, in);
	}

	public void addAtIndex(int i, int j, int k, Complex in) {
		setAtIndex(i, j, k, getAtIndex(i, j, k).add(in));
	}


	public void subtractAtIndex(int i, int j, int k, float val) {
		subtractAtIndex(i, j, k, val, 0);
	}

	public void subtractAtIndex(int i, int j, int k, float real, float imag) {
		Complex in = new Complex(real, imag);
		subtractAtIndex(i, j, k, in);
	}

	public void subtractAtIndex(int i, int j, int k, Complex in) {
		setAtIndex(i, j, k, getAtIndex(i, j, k).sub(in));
	}


	public void setAtIndex(int i, int j, int k, float val){
		setAtIndex(i, j, k, val, 0);
	}

	public void setAtIndex(int i, int j, int k, float real, float imag){
		Complex in = new Complex(real, imag);
		setAtIndex(i, j, k, in);
	}

	public void setAtIndex(int i, int j, int k, Complex val){
		this.getSubGrid(k).getSubGrid(j).setAtIndex(i,(float)val.getReal(),(float)val.getImag());
		notifyAfterWrite();
	}


	public Complex getAtIndex(int i, int j, int k){
		notifyBeforeRead();
		return getSubGrid(k).getSubGrid(j).getAtIndex(i);
	}

	public ComplexGrid2D getSubGrid(int k){
		notifyBeforeRead();
		return subGrids[k];
	}

	public void setSubGrid(int k, ComplexGrid2D slice){
		subGrids[k]=slice;
		notifyAfterWrite();
	}


	@Override
	public Complex getValue(int... idx) {
		return getAtIndex(idx[0],idx[1],idx[2]);
	}

	@Override
	public void setValue(Complex val, int... idx) {
		setAtIndex(idx[0], idx[1], idx[2], val);
	}


	@Override
	public ComplexGrid clone() {
		notifyBeforeRead();
		return new ComplexGrid3D(this);
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
	public Grid3D getRealGrid() {
		return getRealSubGrid(0, 0, 0, size[0], size[1], size[2]);
	}

	@Override
	public Grid3D getImagGrid() {
		return getImagSubGrid(0, 0, 0, size[0], size[1], size[2]);
	}

	@Override
	public Grid3D getMagGrid() {
		return getMagSubGrid(0, 0, 0, size[0], size[1], size[2]);
	}

	@Override
	public Grid3D getPhaseGrid() {
		return getPhaseSubGrid(0, 0, 0, size[0], size[1], size[2]);
	}

	public Grid3D getRealSubGrid(final int startX, final int startY, final int startZ, final int lengthX, final int lengthY, final int lengthZ){
		Grid3D subgrid = new Grid3D(lengthX,lengthY,lengthZ);
		for (int k=0; k < lengthZ; ++k){
			for (int j=0; j < lengthY; ++j){
				for (int i=0; i < lengthX; ++i){
					subgrid.setAtIndex(i, j, k, this.getRealAtIndex(i+startX,j+startY,k+startZ));
				}
			}
		}
		return subgrid;
	}

	public Grid3D getImagSubGrid(final int startX, final int startY, final int startZ, final int lengthX, final int lengthY, final int lengthZ){
		Grid3D subgrid = new Grid3D(lengthX,lengthY,lengthZ);
		for (int k=0; k < lengthZ; ++k){
			for (int j=0; j < lengthY; ++j){
				for (int i=0; i < lengthX; ++i){
					subgrid.setAtIndex(i, j, k, this.getImagAtIndex(i+startX,j+startY,k+startZ));
				}
			}
		}
		return subgrid;
	}

	public Grid3D getMagSubGrid(final int startX, final int startY, final int startZ, final int lengthX, final int lengthY, final int lengthZ){
		Grid3D subgrid = new Grid3D(lengthX,lengthY,lengthZ);
		for (int k=0; k < lengthZ; ++k){
			for (int j=0; j < lengthY; ++j){
				for (int i=0; i < lengthX; ++i){
					float x = this.getRealAtIndex(i+startX,j+startY,k+startZ);
					float y = this.getImagAtIndex(i+startX,j+startY,k+startZ);
					subgrid.setAtIndex(i,j,k, (float)Math.sqrt(x*x+y*y));
				}
			}
		}
		return subgrid;
	}

	public Grid3D getPhaseSubGrid(final int startX, final int startY, final int startZ, final int lengthX, final int lengthY, final int lengthZ){
		Grid3D subgrid = new Grid3D(lengthX,lengthY,lengthZ);
		for (int k=0; k < lengthZ; ++k){
			for (int j=0; j < lengthY; ++j){
				for (int i=0; i < lengthX; ++i){
					float x = this.getRealAtIndex(i+startX,j+startY,k+startZ);
					float y = this.getImagAtIndex(i+startX,j+startY,k+startZ);
					subgrid.setAtIndex(i,j,k, (float)Math.atan2(y,x));
				}
			}
		}
		return subgrid;
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
		return (float)getAtIndex(idx[0],idx[1],idx[2]).getReal();
	}

	@Override
	public void setRealAtIndex(float val, int... idx) {
		setAtIndex(idx[0],idx[1],idx[2], val);
	}

	@Override
	public float getImagAtIndex(int... idx) {
		return (float)getAtIndex(idx[0],idx[1],idx[2]).getImag();
	}

	@Override
	public void setImagAtIndex(float val, int... idx) {
		Complex cval = getAtIndex(idx[0],idx[1],idx[2]);
		cval.setImag(val);
		setAtIndex(idx[0],idx[1],idx[2], cval);
	}

	@Override
	public int getOffset() {
		return 0;
	}
	
	public static void main(String[] args) {
		new ImageJ();
		int pow2 = 226;
		ComplexGrid3D bla = new ComplexGrid3D(pow2,pow2,pow2);
		int max = bla.getNumberOfElements();
		for (int k = 0; k < max; k++) {
			float h = (float)((double)k*Math.sqrt(2)/bla.getNumberOfElements());
			bla.getBuffer()[2*k]=h;
			bla.getBuffer()[2*k+1]=-1*h;
		}


		ComplexGrid3D gc = null;
		ComplexGrid3D gc1 = null;
		ComplexGrid3D gc2 = null;
		long[] ellapsed = new long[3];

		gc = new ComplexGrid3D(bla);
		gc1 = new ComplexGrid3D(gc);
		gc2 = new ComplexGrid3D(gc);
		bla.show("GT");

		Fourier ft = new Fourier();
		
		int reps = 1;
		for (int i = 0; i < reps; i++) {
			
			long start = System.nanoTime();
			ft.fft(gc);
			ft.ifft(gc);
			ellapsed[0] += System.nanoTime() - start;
			start = System.nanoTime();
			ft.fft2(gc1);
			ft.ifft2(gc1);
			ellapsed[1] += System.nanoTime() - start;
			start = System.nanoTime();
			ft.fft3(gc2);
			ft.ifft3(gc2);
			ellapsed[2] += System.nanoTime() - start;
		}

		for (int i = 0; i < ellapsed.length; i++) {
			System.out.println(((double)ellapsed[i])/1e6/(double)reps);
		}

		gc.show("FFT1");
		gc1.show("FFT2");
		gc2.show("FFT3");


	}

}

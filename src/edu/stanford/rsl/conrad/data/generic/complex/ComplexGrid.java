/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;


import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.data.Grid;
import edu.stanford.rsl.conrad.data.generic.GenericGrid;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;


public abstract class ComplexGrid extends GenericGrid<Complex> {

	public abstract float[] getAslinearMemory();

	public abstract void setAslinearMemory(float[] buffer);

	public abstract Grid getRealGrid();

	public abstract Grid getImagGrid();

	public abstract Grid getMagGrid();

	public abstract Grid getPhaseGrid();

	public abstract float getRealAtIndex(int... idx);

	public abstract void setRealAtIndex(float val, int... idx);

	public abstract float getImagAtIndex(int... idx);

	public abstract void setImagAtIndex(float val, int... idx);

	public abstract int getOffset();
	
	@Override
	public void show() {
		show(this.getClass().getSimpleName());
	}

	@Override
	public void show(String s) {
		getMagGrid().show(s+ " - Magnitude");
		//getPhaseGrid().show(s+ " - Phase");
	}

	@Override
	public void initializeDelegate(CLContext context, CLDevice device) {
		if(this.getAslinearMemory()==null) throw new NullPointerException("Host buffer needs to be initialized before the OpenCL delegate can be created");
		delegate = new OpenCLComplexMemoryDelegate(this.getAslinearMemory(), context, device);
	}
	
	@Override
	public ComplexGridOperatorInterface getGridOperator() {
		return (openCLactive) ? OpenCLComplexGridOperator.getInstance() : ComplexGridOperator.getInstance();
	}
	
	@Override
	public ComplexGridOperatorInterface selectGridOperator(boolean useOpenCLOperator) {
		return (useOpenCLOperator) ? new OpenCLComplexGridOperator() : ComplexGridOperator.getInstance();
	}

}

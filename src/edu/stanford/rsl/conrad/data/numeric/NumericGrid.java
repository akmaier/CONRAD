/*
 * Copyright (C) 2010-2014 - Andreas Maier, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.numeric;

import edu.stanford.rsl.conrad.data.Grid;


/**
 * The abstract class for numeric (float) grids
 * 
 * @author Andreas Keil
 */
public abstract class NumericGrid extends Grid{

	protected boolean debug = true;

	/**
	 * @return Returns the value at position idx
	 */
	public abstract float getValue(int[] idx);

	/**
	 * @return Sets the value at position idx
	 */
	public abstract void setValue(float val, int[] idx);

	@Override
	public abstract NumericGrid clone();

	public abstract NumericGrid getSubGrid(int index);
	
	
	// *********************************************************
	// ************** OpenCL related methods *******************
	// *********************************************************
	// TODO: Try to move to OpenCLGridInterface
	// Requires better separation of Grid and OpenCLGrid classes
	
	/*
	 * Does nothing on purpose / No notification for non-CL grids
	 */
	public void notifyBeforeRead(){};

	/*
	 * Does nothing on purpose / No notification for non-CL grids
	 */
	public void notifyAfterWrite(){};	

	/*
	 * Gives the grids corresponding grid operator
	 */
	public NumericGridOperator getGridOperator(){
		return NumericGridOperator.getInstance();
	}
}



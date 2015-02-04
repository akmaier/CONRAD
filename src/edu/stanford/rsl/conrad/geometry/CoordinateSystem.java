package edu.stanford.rsl.conrad.geometry;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

public class CoordinateSystem {
	private Axis [] axis;
		
	public CoordinateSystem(Axis...axis){
		init(axis);
	}
	
	protected void init(Axis [] axis){
		this.axis = axis.clone();
	}

	public Axis [] Axes(){
		return axis;
	}
	
	public void applyChangeOfCoordinatesMatrix(SimpleMatrix  transform){
		
	}
	
	public Axis getAxis(int index){
		return axis[index];
	}
	
	public int dimension(){
		return axis.length;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
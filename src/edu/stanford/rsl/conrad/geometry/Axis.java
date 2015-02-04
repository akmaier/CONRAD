package edu.stanford.rsl.conrad.geometry;


import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to model a coordinate axis;
 * @author Rotimi X Ojo
 */
public class Axis {

	private SimpleVector axis;
	
	public Axis(double ... coordinates){
		axis = new SimpleVector(coordinates).normalizedL2();
	}
	
	public Axis(SimpleVector axisvec){
		axis = axisvec.normalizedL2();
	}
	
	public Axis(Axis a){
		axis = (a.axis != null) ? a.axis.clone() : null;
	}

	/**
	 * @return a unit vector defining an axis
	 */
	public SimpleVector getAxisVector(){
		return axis.clone();
	}
	
	/**
	 * @return the dimension of the axis vector
	 */
	public int dimension() {
		return axis.getLen();
	}

	public void setAxis(SimpleVector newAxis) {
		axis = newAxis.normalizedL2();		
	}	
	
	@Override
	public Axis clone(){
		return new Axis(this);
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes;

import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.CoordinateSystem;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class StandardCoordinateSystem extends CoordinateSystem {
	
	public StandardCoordinateSystem(int dimension){
		Axis [] axis = new Axis[dimension];
		for(int i = 0; i < dimension; i++){
			SimpleVector axisvec = new SimpleVector(dimension);
			axisvec.setElementValue(i, 1);
			axis[i] = new Axis(axisvec);
		}
		init(axis);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;


/**
 * Abstract class to model n dimensional curves.
 * Note that the internal dimension (cf. description by Piegl), i.e. its parametrization is always one dimensional.
 * Hence, a curve can only be evaluated using a single dimension. 
 * That is the reason, why the abstract curve overides the evaluate PointND method.
 * In it only the first dimension of the ND point is used for evaluation of the curve.
 * The abstract method evaluate(double) must be implemented by any derived class that is not abstract to describe how to do the actual evaluation. 
 * 
 * @author akmaier
 *
 *@see AbstractShape
 *@see #evaluate(double)
 */
public abstract class AbstractCurve extends AbstractShape {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8742936166235392950L;

	public AbstractCurve(){
		super();
	}
	
	public AbstractCurve(AbstractCurve ac){
		super(ac);
	}
	
	@Override
	public PointND evaluate(PointND u) {
		return evaluate(u.get(0)); 
	}
	
	/**
	 * Returns a point on the Curve at position u [0, 1];
	 * @param u the internal one dimensional position
	 * @return the curve point
	 */
	public abstract PointND evaluate(double u);

	
	@Override
	public int getInternalDimension() {
		// Internal dimension is always one as this class models a one dimensional abstract shape.
		return 1;
	}

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
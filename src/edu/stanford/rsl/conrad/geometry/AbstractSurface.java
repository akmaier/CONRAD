package edu.stanford.rsl.conrad.geometry;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;


/**
 * Abstract class to model n dimensional surfaces.
 * Note that the internal dimension (cf. description by Piegl), i.e. its parametrization is always two dimensional.
 * Hence, a surface can only be evaluated using a two dimensions. 
 * That is the reason, why the abstract surface overides the evaluate PointND method.
 * In it only the first two dimensions of the ND point are used for evaluation of the surface.
 * The abstract method evaluate(double, double) must be implemented by any derived class that is not abstract to describe how to do the actual evaluation. 
 * 
 * @author akmaier
 *
 *@see AbstractShape
 *@see #evaluate(double, double)
 *
 */
public abstract class AbstractSurface extends AbstractShape {
	
	public AbstractSurface(){
		super();
	}

	public AbstractSurface(AbstractShape shape) {
		super(shape);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 8336440075236081077L;


	@Override
	public PointND evaluate(PointND u) {
		return evaluate(u.get(0), u.get(1));
	}
	
	/**
	 * Returns a point on the surface at position (u, v). u, v in [0, 1];
	 * 
	 * @param u the internal position in u dimension
	 * @param v the internal position in v dimension
	 * @return the surface point
	 */
	public abstract PointND evaluate (double u, double v);

	
	@Override
	public int getInternalDimension() {
		// surfaces do always have the internal dimension two.
		return 2;
	}
	
	/**
	 * Creates a list of connected triangles that can be used to approximate the object. The maximal point to mesh error is given as accuracy.
	 * @param accuracy the maximal deviation in [mm]
	 * @return the compound shape of triangles.
	 */
	public abstract AbstractShape tessellate(double accuracy);
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
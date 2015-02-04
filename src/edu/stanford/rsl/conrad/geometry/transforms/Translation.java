package edu.stanford.rsl.conrad.geometry.transforms;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This class decorates a translation vector. it translates points without altering direction of travel.
 * @author Rotimi X Ojo
 *
 */
public class Translation extends Transform {
	
	private static final long serialVersionUID = 3605382281959086496L;
	private SimpleVector translator = null;
	
	/**
	 * Initialize a translation transform with a translation vector
	 * @param t is vector equivalent of a translation matrix
	 */
	public Translation(SimpleVector t){
		translator = t;
	}
	
	/**
	 * Initialize a translation transform with a translation vector
	 * @param t is vector defined by comma-separated values
	 */
	public Translation(double ... t){
		translator = new SimpleVector(t);
	}
	
	/**
	 * Applies the vector equivalent of a translation matrix on a given point.
	 * @param point is point to be translated.
	 * @return point y = v + point where v is a translation vector.
	 */
	@Override
	public PointND transform(PointND point) {
		return new PointND(SimpleOperators.add(point.getAbstractVector(),translator));
	}
	
	/**
	 * Since  directions are not altered by translation, this method returns a given input unaltered.
	 * @param dir is direction to be translated.
	 * @return dir
	 */
	@Override
	public SimpleVector transform(SimpleVector dir) {
		return dir;
	}

	@Override
	public Translation inverse() {
		return new Translation(translator.negated());		
	}

	@Override
	public Transform clone() {
		return new Translation(translator.clone());
	}

	@Override
	public SimpleVector getData() {
		return translator;
	}
	

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
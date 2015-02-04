package edu.stanford.rsl.conrad.geometry.transforms;

/**
 * Interface for all Objects which can be transformed by Transforms.
 * 
 * @author akmaier
 *
 */
public interface Transformable {
	
	/**
	 * Applies the Transform t to the object.
	 * @param t the transform to apply to the object.
	 */
	public void applyTransform(Transform t);
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.bounds;

import java.io.Serializable;

import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/**
 * Abstract implementation of a bounding condition.
 * 
 * @author Rotimi X Ojo
 *
 */
public abstract class AbstractBoundingCondition implements Serializable {
	
	public AbstractBoundingCondition(){
	}
	
	public AbstractBoundingCondition(AbstractBoundingCondition cond){
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8678860762808220344L;

	/**
	 * Determines whether the point is satisfied by the bounding condition
	 * @param point the point
	 * @return whether the condition is satisfied 
	 */
	public abstract boolean isSatisfiedBy(PointND point);
	
	public abstract AbstractSurface getBoundingSurface();
	
	/**
	 * Invert bounding space.
	 */
	public abstract void flipCondition();
	
	public abstract AbstractBoundingCondition clone();
	
	
}
/*
 * Copyright (C) 2010 - Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
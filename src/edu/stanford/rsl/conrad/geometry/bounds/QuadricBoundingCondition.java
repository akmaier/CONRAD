package edu.stanford.rsl.conrad.geometry.bounds;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.QuadricSurface;

/**
 * This class describes the bounds defined by a {@link QuadricSurface}. This class is useful when a quadric shape describes the extremum of an arbitrary shape
 * @author Rotimi X Ojo
 * @see QuadricSurface
 */
public class QuadricBoundingCondition extends AbstractBoundingCondition{


	public static final long serialVersionUID = -3212222341769199451L;

	private QuadricSurface surface;

	private boolean isFlipped = false;

	/**
	 * Initialize bounding condition defiend by {@link QuadricSurface}.
	 * @param surface is {@link QuadricSurface} describing the extremum of an arbitrary shape.
	 */
	public QuadricBoundingCondition(QuadricSurface surface){
		this.surface = surface;
	}
	
	public QuadricBoundingCondition(QuadricBoundingCondition qbc){
		super(qbc);
		surface = (qbc.surface != null) ? (QuadricSurface) qbc.surface.clone() : null;
		isFlipped = qbc.isFlipped;
	}

	@Override
	public boolean isSatisfiedBy(PointND point) {
		boolean val = surface.isMember(point);
		if(isFlipped){
			return val == true? false: true;
		}			
		return val;
	}

	@Override
	public QuadricSurface getBoundingSurface() {
		return surface;
	}

	@Override
	public void flipCondition() {
		isFlipped = isFlipped==true ? false:true;		
	}

	@Override
	public AbstractBoundingCondition clone() {
		return new QuadricBoundingCondition(this);
	}

}
/*
 * Copyright (C) 2010 - Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
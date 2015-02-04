package edu.stanford.rsl.conrad.geometry.bounds;

import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/**
 * Implementation of a Bounding Box
 * 
 * @author Akm
 * @author Rotimi X Ojo
 *
 */
public class BoundingBox extends AbstractBoundingCondition {
	private static final long serialVersionUID = -5984414347382076217L;
	protected PointND min;
	protected PointND max;
	
	public BoundingBox(PointND min, PointND max){
		this.min = min;
		this.max = max;
	}
	
	public BoundingBox(BoundingBox bb){
		super(bb);
		min = (bb.min != null) ? new PointND(bb.min) : null;
		max = (bb.max != null) ? new PointND(bb.max) : null;
	}
	
	@Override
	public boolean isSatisfiedBy(PointND point) {
		boolean retval = true;
		for (int i=0; i < point.getDimension(); i++){
			if ((point.get(i) > max.get(i)) || (point.get(i) < min.get(i))) retval = false;
		}
		return retval;
	}

	@Override
	public AbstractSurface getBoundingSurface() {
		return null;
	}

	@Override
	public void flipCondition() {
		
	}

	@Override
	public AbstractBoundingCondition clone() {
		return new BoundingBox(this);
	}

}
/*
 * Copyright (C) 2010 - Rotimi X Ojo, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

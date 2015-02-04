package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.Comparator;

import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalPoint;

public class ProjectPointToLineComparator implements Comparator<PhysicalPoint>, Cloneable{

	protected StraightLine projectionLine = null;
	
	public ProjectPointToLineComparator() {
	}
	
	@Override
	public int compare(PhysicalPoint vec1, PhysicalPoint vec2) {
		SimpleVector dir = projectionLine.getDirection();
		return Double.compare(SimpleOperators.multiplyInnerProd(dir, vec1.getAbstractVector()),SimpleOperators.multiplyInnerProd(dir, vec2.getAbstractVector()));
	}

	/**
	 * @return the projectionLine
	 */
	public StraightLine getProjectionLine() {
		return projectionLine;
	}

	/**
	 * @param projectionLine the projectionLine to set
	 */
	public void setProjectionLine(StraightLine projectionLine) {
		this.projectionLine = projectionLine;
	}
	
	@Override
	public ProjectPointToLineComparator clone(){
		ProjectPointToLineComparator clone = new ProjectPointToLineComparator();
		clone.setProjectionLine(projectionLine);
		return clone;
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
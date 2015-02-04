package edu.stanford.rsl.conrad.geometry.shapes.simple;

import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class SortablePoint extends PointND implements Comparable<SortablePoint> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6976938525970166621L;
	private int sortIndex = 0;

	public SortablePoint(double ... d) {
		super(d);
	}
	
	/**
	 * Copy constructor for PointND
	 * @param point
	 */
	public SortablePoint(PointND point) {
		super(point);
	}

	public SortablePoint(SimpleVector add) {
		super(add);
	}

	/**
	 * @return the sortIndex
	 */
	public int getSortIndex() {
		return sortIndex;
	}

	/**
	 * @param sortIndex the sortIndex to set
	 */
	public void setSortIndex(int sortIndex) {
		this.sortIndex = sortIndex;
	}

	@Override
	public int compareTo(SortablePoint o) {
		return Double.compare(this.get(sortIndex), o.get(sortIndex));
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
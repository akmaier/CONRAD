/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.organization;

import java.util.Comparator;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class ComparablePoint2D extends PointND  implements Comparable<ComparablePoint2D>{

	/**
	 * 
	 */
	private static final long serialVersionUID = -6311952076787459617L;


	public static final Comparator<ComparablePoint2D> X_COMPARATOR = new Comparator<ComparablePoint2D>() {
		/**
		 * {@inheritDoc}
		 */
		@Override
		public int compare(ComparablePoint2D o1, ComparablePoint2D o2) {
			if (o1.get(0) < o2.get(0))
				return -1;
			if (o1.get(0) > o2.get(0))
				return 1;
			return 0;
		}
	};

	public static final Comparator<ComparablePoint2D> Y_COMPARATOR = new Comparator<ComparablePoint2D>() {
		/**
		 * {@inheritDoc}
		 */
		@Override
		public int compare(ComparablePoint2D o1, ComparablePoint2D o2) {
			if (o1.get(1) < o2.get(1))
				return -1;
			if (o1.get(1) > o2.get(1))
				return 1;
			return 0;
		}
	};

	public ComparablePoint2D(PointND nonComparablePoint){
		super(nonComparablePoint);
	}


	/*public ComparablePointND(SimpleVector PointND){
    	super(PointND);
    }*/

	public ComparablePoint2D(double... values) {
		super(values);
	}

	public ComparablePoint2D(SimpleVector arrayList){
		super(arrayList);
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null)
			return false;
		if (!(obj instanceof ComparablePoint2D))
			return false;

		ComparablePoint2D point = (ComparablePoint2D) obj;
		return compareTo(point) == 0;
	}

	@Override
	public int compareTo(ComparablePoint2D o) {
		int xComp = X_COMPARATOR.compare(this, o);
		if (xComp != 0)
			return xComp;
		int yComp = Y_COMPARATOR.compare(this, o);
			return yComp;
	}

	/**
	 * Computes the Euclidean distance from this point to the other.
	 * 
	 * @param o1
	 *            other point.
	 * @return euclidean distance.
	 */
	public double euclideanDistance(ComparablePoint2D o1) {
		return euclideanDistance(o1, this);
	}

	/**
	 * Computes the Euclidean distance from one point to the other.
	 * 
	 * @param o1
	 *            first point.
	 * @param o2
	 *            second point.
	 * @return euclidean distance.
	 */
	private static final double euclideanDistance(ComparablePoint2D o1, ComparablePoint2D o2) {
		return General.euclideanDistance(o1.getAbstractVector(), o2.getAbstractVector());
	};


	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("(");
		builder.append(this.get(0)).append(", ");
		builder.append(this.get(1));
		builder.append(")");
		return builder.toString();
	}



}
package edu.stanford.rsl.conrad.angio.graphs.connectedness;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class SkeletonEdge extends Edge{

	private static final long serialVersionUID = 1384271613360467527L;

	public SkeletonEdge(PointND point, PointND point2) {
		super(point, point2);
	}

}

package edu.stanford.rsl.conrad.geometry.shapes.simple;

import edu.stanford.rsl.conrad.physics.PhysicalPoint;

public class LineComparator1D  extends ProjectPointToLineComparator {

	public LineComparator1D() {
	}
	
	@Override
	public int compare(PhysicalPoint vec1, PhysicalPoint vec2) {
		return Double.compare(vec1.getAbstractVector().getElement(0),vec2.getAbstractVector().getElement(0));
	}

	@Override
	public ProjectPointToLineComparator clone(){
		ProjectPointToLineComparator clone = new LineComparator1D();
		clone.setProjectionLine(projectionLine);
		return clone;
	}
	
}

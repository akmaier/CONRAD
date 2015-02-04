package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.ProjectPointToLineComparator;
import edu.stanford.rsl.conrad.geometry.shapes.simple.LineComparator1D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PhysicalPoint;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class Simple1DRayTracer extends SimpleRayTracer {

	public Simple1DRayTracer () {
		comparator = new LineComparator1D();
	}
	
	@Override
	protected ArrayList<PhysicalPoint> intersectWithScene(AbstractCurve ray){
		ArrayList<PhysicalPoint> rayList = new ArrayList<PhysicalPoint>();
		// compute ray intersections:
		for (PhysicalObject shape: scene) {
			if (shape.getShape() == null){
				throw new RuntimeException("Shape " + shape + " did not contain geometric information!");
			}
			if (shape.getShape().getHitsOnBoundingBox(ray).size()>0) {
				ArrayList<PointND> intersection = shape.intersect(ray);
				if (intersection.size() > 0){ 
					for (PointND p : intersection){
						PhysicalPoint point = new PhysicalPoint(p.get(0));
						point.setObject(shape);
						rayList.add(point);
					}
					if(intersection.size() == 1) {
						PhysicalPoint point = new PhysicalPoint(intersection.get(0).get(0)+CONRAD.SMALL_VALUE);
						point.setObject(shape);
						rayList.add(point);
					}
					if(intersection.size() == 3) {
						PhysicalPoint point = new PhysicalPoint(intersection.get(0).get(0)+CONRAD.SMALL_VALUE);
						point.setObject(shape);
						rayList.add(point);
					}
				}
			}
		}
		return rayList;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
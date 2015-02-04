package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;
import java.util.Stack;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.ProjectPointToLineComparator;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PhysicalPoint;

public class SimpleRayTracer extends AbstractRayTracer {

	public SimpleRayTracer () {
		comparator = new ProjectPointToLineComparator();
	}
	
	@Override
	protected ArrayList<PhysicalObject> computeMaterialIntersectionSegments(PhysicalPoint [] rayList){
		ArrayList<PhysicalObject> segments = new ArrayList<PhysicalObject>();
		// determine ray segments

		Stack<PhysicalObject> materialStack = new Stack<PhysicalObject>();
		materialStack.push(rayList[0].getObject());
		for (int k=1; k < rayList.length;k++){
			PhysicalObject obj = new PhysicalObject();
			Edge edge = new Edge(rayList[k-1], rayList[k]);
			obj.setShape(edge);
			if(materialStack.isEmpty()) {
				obj.setMaterial(scene.getBackgroundMaterial());
				obj.setNameString("Background");
			} else {
				PhysicalObject current = materialStack.peek();
				obj.setMaterial(current.getMaterial());
				obj.setNameString(current.getNameString());
			}
			segments.add(obj);
			PhysicalObject nextObject = rayList[k].getObject();
			if (materialStack.contains(nextObject)){
				materialStack.remove(nextObject);
			} else {
				materialStack.push(nextObject);
			}
			
		}


		return segments;
	}



}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
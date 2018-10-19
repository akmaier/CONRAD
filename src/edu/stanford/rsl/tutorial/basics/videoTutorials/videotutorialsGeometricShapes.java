package edu.stanford.rsl.tutorial.basics.videoTutorials;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;

public class videotutorialsGeometricShapes {

	public static void main(String[] args) {
		
		//create a Box between (0,0,0) and (3,4,5)
		Box box = new Box(3.d,2.d,5.d);
		
		//create two PointNDs: (1,-2,0) and (1,-1,2)
		PointND p1 = new PointND(1.d,-2.d,0.d);
		PointND p2 = new PointND(1.d,-1.d,2.d);
		
		//create a StraightLine through these PointNDs
		StraightLine sl = new StraightLine(p1,p2);
		
		//compute the intersection of sl and box
		ArrayList<PointND> intersections = box.intersect(sl);
		
		//check if sl actually intersects box
		if (intersections.size() != 2){
			//if there's no intersection, change the direction of sl
			if(intersections.size() == 0) {
				sl.getDirection().multiplyBy(-1.d);
				intersections = box.intersect(sl);
				System.out.println("Direction changed!");
			}
			//if there's still no intersection, sl passes box
			if(intersections.size() == 0) {
				System.out.println("The StraightLine doesn't intersect the Box!");
				return;
			}
		}
		
		//print the intersections
		System.out.println("Entry point: " + intersections.get(0));
		System.out.println("Exit point: " + intersections.get(1));

	}

}

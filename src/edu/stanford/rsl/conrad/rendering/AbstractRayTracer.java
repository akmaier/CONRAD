package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.compound.TriangleMesh;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.ProjectPointToLineComparator;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.PhysicalPoint;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Abstract Class to model a ray caster. The ray caster casts rays through the scene and determines all his along a ray.
 * Then the ray caster determines the line segments between the objects and determines their representation.
 * 
 * @author akmaier
 *
 */
public abstract class AbstractRayTracer {

	protected AbstractScene scene;
	protected ProjectPointToLineComparator comparator =null;

	/**
	 * Inconsistent data correction will slow down the projection process, but will help to correct for problems in the data like an uneven number of hits of an object.
	 */
	protected boolean inconsistentDataCorrection = true;
	
	// Cache the information whether an object is a triangle or not
	private HashMap<PhysicalObject, Boolean> objIsTriangleCache = new HashMap<>();

	/**
	 * @return the scene
	 */
	public AbstractScene getScene() {
		return scene;
	}

	/**
	 * @param scene the scene to set
	 */
	public void setScene(AbstractScene scene) {
		this.scene = scene;
	}

	/**
	 * Method to cast a ray through the scene. Returns the edge segments which pass through different materials.
	 * <BR><BR>
	 * Rays must be normalized!
	 * <BR>
	 * @param ray
	 * @return the list of line segments which were hit by the ray in the correct order
	 */
	public ArrayList<PhysicalObject> castRay(AbstractCurve ray) {
		ArrayList<PhysicalPoint> rayList = intersectWithScene(ray);
		boolean doubles = false;
		// Filter double hits
		while (doubles){
			boolean foundMore = false;
			int size = rayList.size();
			for (int i = 0; i < size; i++){
				for (int j = i+1; j < size; j++){
					if (rayList.get(i).equals(rayList.get(j))){
						foundMore = true;
						rayList.remove(j);
						size--;
						j--;
					}
				}
			}
			if (!foundMore) doubles = false;
		}
		if (rayList.size() > 0){

			// sort the points along the ray direction in ascending or descending order
			ProjectPointToLineComparator clone = comparator.clone();
			clone.setProjectionLine((StraightLine) ray);
			Collections.sort(rayList, clone);

			// filter consecutive points entering or leaving the object
			rayList = filterDoubles(rayList);
			if (rayList.size() == 0) {
				return null;
			}
			
			PhysicalPoint [] points =  new PhysicalPoint[rayList.size()];
			points = rayList.toArray(points);
			
			
			if (inconsistentDataCorrection){
				ArrayList<PhysicalObject> objects = new ArrayList<PhysicalObject>();
				ArrayList<Integer> count = new ArrayList<Integer>();
				for (int i = 0; i < points.length; i++){
					PhysicalPoint p = points[i];
					boolean found = false;
					for (int j =0; j< objects.size(); j++){
						if (objects.get(j).equals(p.getObject())){
							found = true;
							count.set(j, count.get(j) + 1);
						}			
					}
					if (!found){
						objects.add(p.getObject());
						count.add(new Integer(1));
					}
				}
				boolean rewrite = false;
				for (int i = 0; i < objects.size(); i++){
					if (count.get(i) %2 == 1){
						boolean resolved = false;
						for (int j = 0; j < points.length; j++){
							if (!resolved) {
								if (points[j].getObject().equals(objects.get(i))){
									// only one hit of this object. no problem
									if (count.get(i) == 1){
										points[j] = null;
										rewrite = true;
										resolved = true;
									}
									// three hits in a row of the same object. no problem.
									if (j > 0 && j < points.length-1) {
										if (points[j-1].getObject().equals(objects.get(i)) && (points[j+1].getObject().equals(objects.get(i)))){
											points[j] = null;
											rewrite = true;
											resolved = true;
										}
									}
								}
							}
						}
						if (!resolved){
							// Still not resolved
							// remove center hit
							int toRemove = (count.get(i) + 1) / 2;
							int current = 0;
							for (int j = 0; j < points.length; j++){
								if (points[j].getObject().equals(objects.get(i))){
									current ++;
									if (current == toRemove){
										points[j] = null;
										rewrite = true;
										resolved = true;
									}
								}
							}
						}
					}
					rayList = new ArrayList<PhysicalPoint>();
					if (rewrite){
						for (int j = 0; j < points.length; j++){
							if (points[j] != null){
								rayList.add(points[j]);
							}
						}
						points = new PhysicalPoint[rayList.size()];
						points = rayList.toArray(points);
					}
				}
			}
			
			if (points.length == 0) {
				return null;
			}
			
			return computeMaterialIntersectionSegments(points);

		} else {
			return null;
		}
	}

	/**
	 * Method to resolve the priority of the elements of the scene.
	 * @param rayList
	 * @return the correct line segments ordered according to the specified ray tracing order
	 */
	protected abstract ArrayList<PhysicalObject> computeMaterialIntersectionSegments(PhysicalPoint[] rayList);

	/**
	 * Computes all intersection of the ray with the scene.
	 * @param ray the ray through the scene
	 * @return the intersection points.
	 */
	protected ArrayList<PhysicalPoint> intersectWithScene(AbstractCurve ray){
		ArrayList<PhysicalPoint> rayList = new ArrayList<PhysicalPoint>();
		SimpleVector smallIncrementAlongRay = SimpleOperators.subtract(ray.evaluate(CONRAD.SMALL_VALUE).getAbstractVector(), ray.evaluate(0).getAbstractVector());
		// compute ray intersections:
		for (PhysicalObject shape: scene) {
			if (shape.getShape().getHitsOnBoundingBox(ray).size() > 0) {
				ArrayList<PointND> intersection = shape.intersectWithHitOrientation(ray);
				
				if (intersection != null && intersection.size() > 0){ 
					for (PointND p : intersection){
						
						PhysicalPoint point;
						// Clean coordinates of intersecting points with triangles, as the last coordinate is abused to return the inclination of the triangle
						if (objIsTriangle(shape)) {
							point = cleanTriangleIntersection(p);
						} else {
							point = new PhysicalPoint(p);
						}
						
						point.setObject(shape);
						rayList.add(point);
					}
					if(intersection.size() == 1) {
						PointND p = intersection.get(0);
						PhysicalPoint point;
						
						// When creating an opposing point, use the negative inclination
						if (objIsTriangle(shape)) {
							PhysicalPoint clean = cleanTriangleIntersection(p);
							p = new PointND(clean.getAbstractVector());
							point = new PhysicalPoint(SimpleOperators.add(p.getAbstractVector(), smallIncrementAlongRay));
							point.setHitOrientation(-clean.getHitOrientation());
						} else {
							point = new PhysicalPoint(SimpleOperators.add(p.getAbstractVector(), smallIncrementAlongRay));
						}
						
						point.setObject(shape);
						rayList.add(point);
					}
					else if(intersection.size() == 3) {
						PointND p = intersection.get(1);
						PhysicalPoint point;
						
						// Same as above
						if (objIsTriangle(shape)) {
							PhysicalPoint clean = cleanTriangleIntersection(p);
							p = new PointND(clean.getAbstractVector());
							point = new PhysicalPoint(SimpleOperators.add(p.getAbstractVector(), smallIncrementAlongRay));
							point.setHitOrientation(-clean.getHitOrientation());
						} else {
							point = new PhysicalPoint(SimpleOperators.add(p.getAbstractVector(), smallIncrementAlongRay));
						}
						
						point.setObject(shape);
						rayList.add(point);
					}
				}
			}
		}
		return rayList;
	}
	

	/**
	 * When we hit a triangle, we store the inner product between the direction of the ray and the normal of the triangle to determine whether we just entered or left an object
	 * Remove the point's abused coordinate and store this information in the physical point object
	 * @param triangleHit point of intersection between ray and triangle. The point's last coordinate isn't a real coordinate
	 * @return physical point with clean coordinates, the hit orientation and the object
	 */
	private PhysicalPoint cleanTriangleIntersection(PointND triangleHit) {
		double[] coordinates = triangleHit.getCoordinates();
		double[] cleanCoordinates = Arrays.copyOf(coordinates, coordinates.length - 1);
		PhysicalPoint point = new PhysicalPoint(new PointND(cleanCoordinates));
		point.setHitOrientation(coordinates[coordinates.length - 1]);
		return point;
	}
	
	
	/**
	 * Determines whether the given object is a triangle or is composed out of (and only out of) triangles
	 * The result is stored in a cache-like hash map such that the result can be quickly retrieved from the map
	 * @param obj to check
	 * @return true if the given object is a triangle, a triangle mesh or a compound shape which does not hold other objects than triangles
	 */
	private boolean objIsTriangle(PhysicalObject obj) {
		
		// First, try to retrieve the cached information
		if (objIsTriangleCache.containsKey(obj)) {
			return objIsTriangleCache.get(obj);
		}
		
		AbstractShape shape = obj.getShape();
		if (shape instanceof Triangle || shape instanceof TriangleMesh) {
			objIsTriangleCache.put(obj, true);
			return true;
		}
		
		// do breadth first search on nested compound shapes
		else if (shape instanceof CompoundShape) {
			LinkedList<AbstractShape> queue = new LinkedList<>();
			queue.add(shape);
			
			while(queue.size() > 0) {
				CompoundShape cs = (CompoundShape) queue.poll();
				
				for (int i=0; i<cs.getInternalDimension(); i++) {
					shape = cs.get(i);
					if (shape instanceof Triangle || shape instanceof TriangleMesh) {
						continue;
					} else if (shape instanceof CompoundShape) {
						queue.add(shape);
					} else {
						objIsTriangleCache.put(obj, false);
						return false;
					}
				}
			}
			
			// if all the nested compound shapes do not hold more than triangles in the end, we're certainly dealing with a triangle
			objIsTriangleCache.put(obj, true);
			return true;
		}
		
		// otherwise it is another shape and thus not a triangle
		objIsTriangleCache.put(obj, false);
		return false;
	}
	
	
	/**
	 * A ray is supposed to enter and leave a closed object composed out of triangles in an alternating order. If the ray e.g. enters the object twice, it's likely that the two intersections considered as entry are very close.
	 * An intersecting point is considered an entry if the dot product of the ray's direction and the hit triangle's normal vector is smaller than 0
	 * @param rayList list of points where the ray intersects with an object
	 * @return
	 */
	private ArrayList<PhysicalPoint> filterDoubles(ArrayList<PhysicalPoint> rayList) {
		ArrayList<PhysicalPoint> filtered = new ArrayList<>();
		
		// First intersection enters the object
		boolean nextEntersObject = true;
		boolean init = false;
		for (int i=0; i<rayList.size(); i++) {
			PhysicalPoint p = rayList.get(i);
			
			// Only for triangles
			if (!objIsTriangle(p.getObject())) {
				filtered.add(p);
				continue;
			}
			
			if (!init) {
				nextEntersObject = !(p.getHitOrientation() < 0);
				filtered.add(p);
				init = true;
			}
			else if (nextEntersObject && p.getHitOrientation() < 0) {
				// Ray enters the object
				filtered.add(p);
				nextEntersObject = false;	// alternate
			}
			else if (!nextEntersObject && p.getHitOrientation() > 0) {
				// Ray leaves the object
				filtered.add(p);
				nextEntersObject = true;	// alternate
			}
			else if (p.getHitOrientation() == 0) {
				// Ray is parallel to the triangle's surface
				filtered.add(p);
			}
			
			// if we come to here, the point is discarded
		}
		
		return filtered;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
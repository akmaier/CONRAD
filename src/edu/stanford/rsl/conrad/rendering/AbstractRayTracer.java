package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.ProjectPointToLineComparator;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
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
			PhysicalPoint [] points =  new PhysicalPoint[rayList.size()];

			// sort the points along the ray direction in ascending or descending order
			ProjectPointToLineComparator clone = comparator.clone();
			clone.setProjectionLine((StraightLine) ray);
			Collections.sort(rayList, clone);
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
			if (shape.getShape().getHitsOnBoundingBox(ray).size()>0) {
				ArrayList<PointND> intersection = shape.intersect(ray);
				if (intersection != null && intersection.size() > 0){ 
					for (PointND p : intersection){
						PhysicalPoint point = new PhysicalPoint(p);
						point.setObject(shape);
						rayList.add(point);
					}
					if(intersection.size() == 1) {
						PhysicalPoint point = new PhysicalPoint(SimpleOperators.add(intersection.get(0).getAbstractVector(), smallIncrementAlongRay));
						point.setObject(shape);
						rayList.add(point);
					}
					else if(intersection.size() == 3) {
						PhysicalPoint point = new PhysicalPoint(SimpleOperators.add(intersection.get(1).getAbstractVector(), smallIncrementAlongRay));
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
package edu.stanford.rsl.conrad.geometry;

import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Transformable;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Class to model any kind of curve or surface.
 * <br><br>
 * <img alt="Saddle Surface" height="150" width="150" src="http://upload.wikimedia.org/wikipedia/commons/2/21/Saddle_pt.jpg"><BR>
 * Example for a 3D surface.
 * <br> 
 * @author akmaier
 *
 */
public abstract class AbstractShape implements Serializable, Transformable {

	private Plane3D [] boundingPlanes;
	protected PointND min;
	protected PointND max;
	private String name;


	public AbstractShape() {
	}

	/**
	 * Copy constructor (deep copy)
	 * @param shape
	 */
	public AbstractShape(AbstractShape shape){
		min = (shape.min != null) ? shape.min.clone() : null;
		max = (shape.max != null) ? shape.max.clone() : null;
		name = (shape.name != null) ? new String(shape.name) : null;
		
		if (shape.boundingPlanes!=null){
			boundingPlanes = new Plane3D[shape.boundingPlanes.length];
			for (int i = 0; i < boundingPlanes.length; i++) {
				boundingPlanes[i] = (shape.boundingPlanes[i]!=null) ? new Plane3D(shape.boundingPlanes[i]) : null;
			}
		}
		else
			boundingPlanes=null;
	}



	/**
	 * Returns a deep copy of this shape
	 * @return Deep copy of this abstract shape
	 */
	public abstract AbstractShape clone();


	/**
	 * Returns true if the shape is of limited space
	 * @return  Boundedness of this shape.
	 */
	public abstract boolean isBounded();

	/**
	 * Evaluates the bounding box and returns true if it is hit. If the object is not bounded, it returns true as default.
	 * 
	 * @param curve the curve
	 * @return true, if the object is hit.
	 */
	public ArrayList<PointND> getHitsOnBoundingBox_slow(AbstractCurve curve){
		if (isBounded()){
			ArrayList<PointND> hits = new ArrayList<PointND>();
			for(int i = 0; i< boundingPlanes.length; i++){
				Plane3D p = boundingPlanes[i];
				try{
					ArrayList<PointND> points = p.intersect(curve);
					for (PointND intersection : points){
						//System.out.println(intersection);
						boolean inBound = true;
						for (int j=0; j < intersection.getDimension(); j++){
							double coord = intersection.get(j);
							if (coord > max.get(j) + CONRAD.SMALL_VALUE || coord < min.get(j) - CONRAD.SMALL_VALUE) {
								inBound = false;
								break;
							}
						}
						if (inBound) {
							hits.add(intersection);
						}
					}
				} catch (RuntimeException e){
					if (!e.getMessage().equals("Line is parallel to plane")) { 
						System.out.println(e.getLocalizedMessage());
					} else {

					}
				} catch (Exception e){
					e.printStackTrace();
				}
			}
			return hits;
		} else {
			throw new RuntimeException("Object is not bounded");
		}
	}

	protected void generateBoundingPlanes(PointND min, PointND max){
		this.min = min;
		this.max = max;
		generateBoundingPlanes();
	}

	public ArrayList<PointND> getHitsOnBoundingBox(AbstractCurve curve){
		if (isBounded()){
			ArrayList<PointND> hits = null;
			if (curve instanceof StraightLine) {
				StraightLine line = (StraightLine) curve;
				hits = General.intersectRayWithCuboid(line, min, max);
			} else {
				hits = new ArrayList<PointND>();
				for(int i = 0; i< boundingPlanes.length; i++){
					Plane3D p = boundingPlanes[i];
					try{
						ArrayList<PointND> points = p.intersect(curve);
						for (PointND intersection : points){
							//System.out.println(intersection);
							boolean inBound = true;
							for (int j=0; j < intersection.getDimension(); j++){
								double coord = intersection.get(j);
								if (coord > max.get(j) + CONRAD.SMALL_VALUE || coord < min.get(j) - CONRAD.SMALL_VALUE) {
									inBound = false;
									break;
								}
							}
							if (inBound) {
								hits.add(intersection);
							}
						}
					} catch (RuntimeException e){
						if (!e.getMessage().equals("Line is parallel to plane")) { 
							System.out.println(e.getLocalizedMessage());
						} else {

						}
					} catch (Exception e){
						e.printStackTrace();
					}
				}
			}
			return hits;
		} else {
			throw new RuntimeException("Object is not bounded");
		}
	}

	protected void generateBoundingPlanes(){
		boundingPlanes = new Plane3D[6];
		//PointND e1 = new PointND(1, 0, 0);
		//PointND e2 = new PointND(0, 1, 0);
		//PointND e3 = new PointND(0, 0, 1);
		//boundingPlanes[0] = new Plane3D(min, e1.getAbstractVector());
		//boundingPlanes[1] = new Plane3D(min, e2.getAbstractVector());
		//boundingPlanes[2] = new Plane3D(min, e3.getAbstractVector());
		//boundingPlanes[3] = new Plane3D(max, e1.getAbstractVector());
		//boundingPlanes[4] = new Plane3D(max, e2.getAbstractVector());
		//boundingPlanes[5] = new Plane3D(max, e3.getAbstractVector());
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 6333594201718418159L;

	/**
	 * Returns the point on the shape at the internal position u. If the shape is a curve, u is of dimension 1, if the shape is a surface u is of dimension 2, etc.
	 * @param u the point in the internal parameter dimension
	 * @return the point on the shape at the internal dimension
	 */
	abstract public PointND evaluate(PointND u);

	/**
	 * Returns the external dimension of the shape.
	 * @return the dimension
	 */
	abstract public int getDimension();

	/**
	 * returns the internal dimension of the shape, i.e. 1 if it is a curve, 2 if it is a surface, etc.
	 * @return the internal dimension
	 */
	abstract public int getInternalDimension();

	/**
	 * Returns the intersection points between the curve and the shape. 
	 * Returns null, if the intersection is empty.
	 * 
	 * @param other
	 * @return the intersection points.
	 */
	abstract public ArrayList<PointND> intersect(AbstractCurve other);
	
	
	/**
	 * For most objects this method simply calls intersect(other);
	 * For triangles the orientation of the hit (scalar product of ray and triangle's normal vectors) is stored as an additional coordinate
	 * 
	 * @param other
	 * @return the intersection points
	 */
	public ArrayList<PointND> intersectWithHitOrientation(AbstractCurve other) {
		return intersect(other);
	}

	/**
	 * Rasters the shape with a given number of points or less. If the shape is not bounded null is returned.
	 * @param number the number of points
	 * @return the raster points
	 */
	abstract public PointND[] getRasterPoints(int number);


	/**
	 * @return the minimal corner of the bounding box.
	 */
	public PointND getMin() {
		return min;
	}


	/**
	 * @return the maximal corner of the bounding box.
	 */
	public PointND getMax() {
		return max;
	}


	public abstract void applyTransform(Transform t);

	public String getName() {
		return name;
	}

	public void setName(String name){
		this.name = name;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.io.Serializable;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Transformable;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Class to model points of arbitrary dimension. Compatible with numerics.
 * 
 * @author akmaier
 *
 */
public class PointND implements Serializable, Transformable {

	protected SimpleVector coordinates;
	/**
	 * @param coordinates the coordinates to set
	 */
	public void setCoordinates(SimpleVector coordinates) {
		this.coordinates = coordinates;
	}


	/**
	 * 
	 */
	private static final long serialVersionUID = -1747968959310910786L;

	/**
	 * Creates a new point of the specified dimension
	 * @param knotVector the dimension
	 */
	public PointND(SimpleVector knotVector){
		coordinates = knotVector;
	}

	public PointND(){

	}

	/**
	 * Copy constructor
	 * @param point
	 */
	public PointND(PointND point){
		coordinates = point.coordinates.clone();
	}

	public PointND clone(){
		return new PointND(coordinates.clone());
	}

	/**
	 * Creates a new point of a given array or list of double values.<br>
	 * <pre>
	 * 	PointND p = new PointND(1.5, 2.5);
	 *	ArbitrayrPoint p2 = new PointND(new double [] {1.5, 2.5});
	 * </pre>
	 * will create two 2D points at the same location.<br>
	 * 
	 * @param point
	 */
	public PointND(double ... point) {
		coordinates = new SimpleVector(point);
	}

	/**
	 * Returns a copy of the point as double array.
	 * @return the coordinates as double []
	 */
	public double [] getCoordinates(){
		double [] revan = new double [coordinates.getLen()];
		coordinates.copyTo(revan);
		return revan;
	}

	/**
	 * Method to retrieve coordinate entries
	 * @param i the index of the coordinate [0, dim[
	 * @return the coordinate
	 */
	public double get(int i){
		return coordinates.getElement(i);
	}

	/**
	 * Methods to set individual entries of the coordinates
	 * @param i the index of the coordinate [0, dim[
	 * @param d the value of the coordinate
	 */
	public void set(int i, double d){
		coordinates.setElementValue(i, d);
	}

	/**
	 * Returns the internal abstract vector to enable computations via the numerics library.<BR>
	 * <b>Changes to the vector will affect the point</b>
	 * @return the internal abstract vector
	 */
	public SimpleVector getAbstractVector(){
		return coordinates;
	}

	/**
	 * Returns the dimension of the point
	 * @return the dimension
	 */
	public int getDimension() {
		return coordinates.getLen();
	}

	/**
	 * computes the Euclidean distance between the current point {@latex.inline $\\mathbf{x_1} = (x_1, y_1)$} the the point "two" {@latex.inline $\\mathbf{x_2} = (x_2, y_2)$} as
	 * {@latex.ilb 
	 		$|| \\mathbf{x_1} - \\mathbf{x_2} ||  = \\sqrt{\\sum_n(x_1(n) - x_2(n))^2}$.
	  }
	 * @param two the other point
	 * @return the distance
	 */
	public double euclideanDistance(PointND two){
		return General.euclideanDistance(this.coordinates, two.coordinates);
		//		assert(two.getDimension() == getDimension());
		//		double revan = 0;
		//		double [] other = two.getCoordinates();
		//		for (int i=0; i< coordinates.getLen(); i++){
		//			revan += Math.pow(other[i] - coordinates.getElement(i), 2);
		//		}
		//		return Math.sqrt(revan);
	}

	/**
	 * Updates the vector at entries which are higher in p. Used to scan for a maximum in an iterative process.
	 * Search for a bounding box can be performed like this:
	 * <pre>
	 * ArrayList<PointND> list = ...
	 * PointND max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
	 * for (PointND p : list){
	 * 	max.updateIfHigher(p);
	 * }
	 * <pre>
	 * 
	 * @param p the other point
	 */
	public void updateIfHigher(PointND p) {
		if (p == null)
			return;

		for (int i = 0; i < coordinates.getLen(); i++) {
			if (p.coordinates.getElement(i) > coordinates.getElement(i)) {
				coordinates.setElementValue(i, p.coordinates.getElement(i));
			}
		}
	}

	/**
	 * Updates the vector at entries which are higher in p. Used to scan for a minimum in an iterative process.
	 * Search for a bounding box can be performed like this:
	 * <pre>
	 * ArrayList<PointND> list = ...
	 * PointND min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
	 * for (PointND p : list){
	 * 	min.updateIfLower(p);
	 * }
	 * <pre>
	 * 
	 * @param p the other point
	 */
	public void updateIfLower(PointND p) {
		if (p == null)
			return;

		for (int i = 0; i < coordinates.getLen(); i++) {
			if (p.coordinates.getElement(i) < coordinates.getElement(i)) {
				coordinates.setElementValue(i, p.coordinates.getElement(i));
			}
		}
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof PointND) {
			PointND p = (PointND) o;
			return (SimpleOperators.subtract(coordinates, p.coordinates).normL2() < CONRAD.FLOAT_EPSILON);
		} else {
			return false;
		}
	}

	@Override
	public void applyTransform(Transform t) {
		coordinates = t.transform(this).coordinates;
	}

	public double innerProduct(PointND pointND) { // suwest
		double innerProduct = 0.0; 
		for(int i = 0; i < pointND.getDimension(); ++i){
			innerProduct += this.get(i) * pointND.get(i);
		}
		return innerProduct;

	}

	/////////////////////////////////////////////////
	// Serialization and Persistence               //
	/////////////////////////////////////////////////

	public String getVectorSerialization() {
		return (coordinates != null) ? coordinates.toString() : null;
	}

	public void setVectorSerialization(final String str) {
		if(str == null) coordinates = null;
		else{
			if (coordinates == null)
				coordinates = new SimpleVector();
			coordinates.init(str);
		}
	}

	@Override
	public String toString(){
		return (coordinates != null) ? coordinates.toString() : null;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
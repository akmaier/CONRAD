package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractSurface;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.bounds.AbstractBoundingCondition;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * <p> CONRAD Class to model surfaces that can efficiently determine the membership of an arbitrary point.
 * <p> The main features provided by this class are point and curve membership verification and bounds checking.
 * 
 *
 * @author Rotimi X Ojo
 *
 */
public abstract class SimpleSurface extends AbstractSurface {
	private static final long serialVersionUID = 1165067022561672962L;
	protected Transform transform;
	protected ArrayList<AbstractBoundingCondition> boundingConditions = new ArrayList<AbstractBoundingCondition>();

	public SimpleSurface(){
		super();
	}
	
	public SimpleSurface(SimpleSurface surf){
		super(surf);
		
		// deep copy the bounding conditions list
		if (surf.boundingConditions != null){
			Iterator<AbstractBoundingCondition> it = surf.boundingConditions.iterator();
			boundingConditions = new ArrayList<AbstractBoundingCondition>();
			while (it.hasNext()) {
				AbstractBoundingCondition cond = it.next();
				boundingConditions.add((cond!=null) ? cond.clone() : null);
			}
		}
		else {
			boundingConditions = null;
		}
		
		// copy the underlying transform
		transform = (surf.transform != null) ? surf.transform.clone() : null;

	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		StraightLine buff = (StraightLine) other;
		StraightLine line = new StraightLine(buff.getPoint().clone(), buff.getDirection().clone());
		line.applyTransform(transform.inverse());		
		ArrayList<PointND> hitsOnShape = getHits(line);
		if(hitsOnShape.size() > 0 && boundingConditions.size() > 0){
			hitsOnShape = removeOBHits(hitsOnShape, line);
			hitsOnShape = addBoundingHits(hitsOnShape, line);
		}
		return getCorrectedHits(hitsOnShape);		
	}


	public abstract ArrayList<PointND> getHits(AbstractCurve other);


	/**
	 * After having obtained the object hits we need to add the bounding condition hits
	 * @param hitsOnBaseShape
	 * @param other
	 * @return
	 */
	protected ArrayList<PointND> addBoundingHits(ArrayList<PointND> hitsOnBaseShape, AbstractCurve other){
		Iterator<AbstractBoundingCondition> boundingclips = boundingConditions.iterator();
		while (boundingclips.hasNext()) {
			AbstractBoundingCondition currClip = boundingclips.next();
			ArrayList<PointND> hitsarr = null;
			try {
				hitsarr = currClip.getBoundingSurface().intersect(other);
			} catch (Exception e) {
				continue;
			}
			Iterator<PointND> cliphits = hitsarr.iterator();
			while (cliphits.hasNext()) {
				PointND currHit = cliphits.next();
				if (isMember(currHit, true)) {
					hitsOnBaseShape.add(currHit);
				}
			}
		}		
		return hitsOnBaseShape;
	}
	
	
	/**
	 * Remove out of bounds hits
	 * @param hitsOnBaseShape
	 * @param other
	 * @return the list without out-of-bound hits
	 */
	protected ArrayList<PointND> removeOBHits(ArrayList<PointND> hitsOnBaseShape, AbstractCurve other) {
		Iterator<PointND> baseHits =  hitsOnBaseShape.iterator();
		while(baseHits.hasNext()){
			PointND currHit = baseHits.next();
			Iterator<AbstractBoundingCondition> boundingclips = boundingConditions.iterator();
			while(boundingclips.hasNext()){
				if(!boundingclips.next().isSatisfiedBy(currHit)){
					baseHits.remove();
					break;
				}
			}
		}
		return hitsOnBaseShape;
	}

	/**
	 * Calculates matrix for rotating shape from a principal axis to new axis.
	 * Special case of change of coordinate.
	 * @param newAxis
	 * @return the matrix
	 */
	public SimpleMatrix getChangeOfAxisMatrix(Axis newAxis) {
		if(newAxis == null || General.areColinear(getPrincipalAxis().getAxisVector(), newAxis.getAxisVector(), CONRAD.FLOAT_EPSILON)){
			SimpleMatrix rotator = new SimpleMatrix(getPrincipalAxis().dimension(),getPrincipalAxis().dimension());
			rotator.identity();
			return rotator;
		}

		SimpleVector Z = newAxis.getAxisVector();
		SimpleVector Y = new Axis(General.crossProduct(Z, getPrincipalAxis().getAxisVector())).getAxisVector();
		SimpleVector X = General.crossProduct(Z, Y);

		double [][] data = {X.copyAsDoubleArray(),Y.copyAsDoubleArray(),Z.copyAsDoubleArray()};
		return new SimpleMatrix(data).transposed();
	}

	/**
	 * Transforms given point (usually from object space to world space).
	 * @param hits hits to be transformed by tranformer.
	 * @return ArrayList of hits in new space
	 */
	protected ArrayList<PointND> getCorrectedHits(ArrayList<PointND> hits){
		ArrayList<PointND> correctedHits = new ArrayList<PointND>(2);
		Iterator<PointND> quadHits = hits.iterator();

		while (quadHits.hasNext()) {
			PointND val = transform.transform(quadHits.next());
			correctedHits.add(val);
		}		

		return correctedHits;
	}

	/**
	 * Determines if the given point in within the bounds of shape;
	 * @param point The point needs to be non-transformed, hence a world coordinate system point
	 * @return true if the point is within the surface
	 */
	public boolean isMember(PointND point){
		return isMember(point, false);
	}
	
	/**
	 * Determines if the given point in within the bounds of shape;
	 * @param point
	 * @param pointTransformed if true, the point is already in a transformed state, otherwise it needs to be transformed
	 * @return true if the point is within the surface
	 */
	public abstract boolean isMember(PointND point, boolean pointTransformed);

	/**
	 * Adds a bounding condition to the top of the "stack". 
	 * @param condition
	 */
	public void addBoundingCondition(AbstractBoundingCondition condition){
		boundingConditions.add(0,condition);
	}


	public abstract Axis getPrincipalAxis();

	/**
	 * Adds a collection of bounding conditions
	 * @param conditions
	 */
	public void addAllBoundingConditions(Collection<? extends AbstractBoundingCondition> conditions){
		boundingConditions.addAll(conditions);
	}

	@Override
	public void applyTransform(Transform t) {
		transform = new ComboTransform(transform, t);
		// transform the eight corners:
		PointND [] corners = new PointND[8];
		corners[0] = new PointND(min.get(0), min.get(1), min.get(2));
		corners[1] = new PointND(min.get(0), min.get(1), max.get(2));
		corners[2] = new PointND(min.get(0), max.get(1), min.get(2));
		corners[3] = new PointND(min.get(0), max.get(1), max.get(2));
		corners[4] = new PointND(max.get(0), min.get(1), min.get(2));
		corners[5] = new PointND(max.get(0), min.get(1), max.get(2));
		corners[6] = new PointND(max.get(0), max.get(1), min.get(2));
		corners[7] = new PointND(max.get(0), max.get(1), max.get(2));
		min = new PointND(Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE);
		max = new PointND(-Double.MAX_VALUE,-Double.MAX_VALUE,-Double.MAX_VALUE);
		for (int i=0; i < 8; i++){
			PointND transf = t.transform(corners[i]);
			min.updateIfLower(transf);
			max.updateIfHigher(transf);
		}
	}

	@Override
	public int getInternalDimension() {
		return 0;
	}

	@Override
	public PointND evaluate(PointND u) {
		return null;
	}

	public float[] getRasterPoints(int elementCountU, int elementCountV) {
		return null;
	}

	/**
	 * Returns cloned copy of shape's affine transform
	 * @return the transform
	 */
	public Transform getTransform() {	
		return transform.clone();
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
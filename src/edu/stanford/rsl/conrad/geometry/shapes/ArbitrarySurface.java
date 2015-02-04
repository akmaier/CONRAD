package edu.stanford.rsl.conrad.geometry.shapes;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.bounds.AbstractBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SimpleSurface;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;

/**
 * Models an arbitrary shape centered at the origin using a base shape and bounding conditions
 * An affine transform class is used to translate and orient shapes from object space in world space
 * The affine transform of an arbitrary shape is always the same as that of its base shape;
 * 
 * @author Rotimi X Ojo
 */
public class ArbitrarySurface extends SimpleSurface {

	private static final long serialVersionUID = -3286378502393392770L;
	private ArrayList<AbstractBoundingCondition> boundingClips = new ArrayList<AbstractBoundingCondition>();
	private SimpleSurface baseSurface;

	public ArbitrarySurface(SimpleSurface baseSurface,Collection<? extends AbstractBoundingCondition> clipSurfaces){
		this.baseSurface = baseSurface;
		this.boundingClips.addAll(boundingClips);
		transform = baseSurface.getTransform();
	}

	public ArbitrarySurface(ArbitrarySurface surf) {
		super(surf);
		
		// deep copy the bounding clips
		if (surf.boundingClips != null){
			Iterator<AbstractBoundingCondition> it = surf.boundingClips.iterator();
			boundingClips = new ArrayList<AbstractBoundingCondition>();
			while (it.hasNext()) {
				AbstractBoundingCondition cond = it.next();
				boundingClips.add((cond!=null) ? cond.clone() : null);
			}
		}
		else {
			boundingClips = null;
		}
		
		// deep copy the base surface
		baseSurface = (surf.baseSurface!=null) ? (SimpleSurface) surf.baseSurface.clone() : null;
	}

	@Override
	public boolean isMember(PointND hit, boolean pointTransformed) {
		if(!baseSurface.isMember(hit, pointTransformed)){
			return false;
		}

		Iterator<AbstractBoundingCondition> clips = boundingClips.iterator();
		while(clips.hasNext()){
			if(!clips.next().isSatisfiedBy(hit)){
				return false;
			}
		}		
		return true;		
	}

	public ArrayList<PointND> getHits(AbstractCurve other){
		return baseSurface.intersect(other);
	}


	@Override
	public boolean isBounded() {
		return true;
	}

	@Override
	public PointND evaluate(PointND u) {
		return u;
	}

	@Override
	public int getDimension() {
		return baseSurface.getDimension();
	}

	@Override
	public int getInternalDimension() {
		return baseSurface.getInternalDimension();
	}


	@Override
	public void applyTransform(Transform t) {
		baseSurface.applyTransform(t);
		transform = baseSurface.getTransform();
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		return null;
	}

	@Override
	public Axis getPrincipalAxis() {
		return baseSurface.getPrincipalAxis();
	}

	@Override
	public PointND evaluate(double u, double v) {
		return baseSurface.evaluate(u, v);
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractShape clone() {
		return new ArbitrarySurface(this);
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
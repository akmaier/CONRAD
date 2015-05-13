package edu.stanford.rsl.conrad.geometry.shapes.compound;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;

public class CompoundShape extends AbstractShape implements Collection<AbstractShape> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4296470814160586436L;
	private ArrayList<AbstractShape> list;
	protected boolean dirty = true;

	public CompoundShape(){
		list = new ArrayList<AbstractShape>();
	}
	
	public CompoundShape(CompoundShape cs){
		super(cs);
		dirty = cs.dirty;
		
		if(cs.list != null){
			Iterator<AbstractShape> it = cs.list.iterator();
			list = new ArrayList<AbstractShape>();
			while (it.hasNext()) {
				AbstractShape shape = it.next();
				list.add((shape!=null) ? shape.clone() : null);
			}
		}
		else
			list = null;
	}

	
	@Override
	public int getDimension() {
		return list.get(0).getDimension();
	}

	public AbstractShape get(int i){
		return list.get(i);
	}

	protected synchronized void init(){
		if (dirty){
			min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
			max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
			for (AbstractShape t: list){
				min.updateIfLower(t.getMin());
				max.updateIfHigher(t.getMax());
			}
			super.generateBoundingPlanes();
			dirty = false;
		}
	}

	@Override
	public ArrayList<PointND> getHitsOnBoundingBox(AbstractCurve curve){
		if (dirty){
			init();
		}
		return super.getHitsOnBoundingBox(curve);
	}

	protected static ArrayList<PointND> intersect(AbstractShape t, AbstractCurve other){
		ArrayList<PointND> pts = null;
		if (t instanceof CompoundShape){
			CompoundShape compound = (CompoundShape) t;
			if (compound.getHitsOnBoundingBox(other).size() > 0){
				pts = compound.intersect(other);
			} else {
				pts = new ArrayList<PointND>();
			}
		} else {
			 pts = t.intersect(other);
		}
		return pts;
	}
	
	protected static ArrayList<PointND> intersectWithHitOrientation(AbstractShape t, AbstractCurve other){
		ArrayList<PointND> pts = null;
		if (t instanceof CompoundShape){
			CompoundShape compound = (CompoundShape) t;
			if (compound.getHitsOnBoundingBox(other).size() > 0){
				pts = compound.intersectWithHitOrientation(other);
			} else {
				pts = new ArrayList<PointND>();
			}
		} else {
			 pts = t.intersectWithHitOrientation(other);
		}
		return pts;
	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		if (dirty){
			init();
		}
		ArrayList <PointND> hits = new ArrayList<PointND>();
		for(AbstractShape t: list){
			hits.addAll(intersect(t,other));
		}
		return hits;
	}
	
	

	@Override
	public ArrayList<PointND> intersectWithHitOrientation(AbstractCurve other) {
		if (dirty){
			init();
		}
		ArrayList <PointND> hits = new ArrayList<PointND>();
		for(AbstractShape t: list){
			hits.addAll(intersectWithHitOrientation(t, other));
		}
		return hits;
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	@Override
	public PointND getMax(){
		if (dirty){
			init();
		}
		return super.getMax();
	}

	@Override
	public PointND getMin(){
		if (dirty){
			init();
		}
		return super.getMin();
	}

	public boolean add(AbstractShape shape){
		boolean revan = list.add(shape);
		dirty = true;
		return revan;
	}

	public PointND[] getRasterPoints(int number){
		ArrayList<PointND[]> lists = new ArrayList<PointND[]>();
		int sum = 0;
		for (AbstractShape t : list){
			PointND [] pts = t.getRasterPoints(number/ list.size());
			sum += pts.length;
			lists.add(pts);
		}
		PointND [] points = new PointND[sum];
		int increment = 0;
		for (PointND [] pts : lists){
			System.arraycopy(pts, 0, points, increment, pts.length);
			increment += pts.length;
		}
		return points;
	}

	@Override
	public void applyTransform(Transform t) {
		for (AbstractShape s: list){
			//System.out.println("CompundShape: applying Transform to " + s.toString());
			s.applyTransform(t);
			//System.out.println("CompundShape: applied Transform to " + s.toString());
		}
		dirty = true;
	}

	@Override
	public PointND evaluate(PointND u) {
		return null;
	}

	@Override
	public int getInternalDimension() {
		return list.size();
	}

	@Override
	public boolean addAll(Collection<? extends AbstractShape> c) {
		dirty = true;
		return list.addAll(c);
	}

	@Override
	public void clear() {
		list.clear();
		dirty = true;
	}

	@Override
	public boolean contains(Object o) {
		return list.contains(o);
	}

	@Override
	public boolean containsAll(Collection<?> c) {
		return list.containsAll(c);
	}

	@Override
	public boolean isEmpty() {
		return list.isEmpty();
	}

	@Override
	public Iterator<AbstractShape> iterator() {
		return list.iterator();
	}

	@Override
	public boolean remove(Object o) {
		dirty = true;
		return list.remove(o);
	}

	@Override
	public boolean removeAll(Collection<?> c) {
		dirty =true;
		return removeAll(c);
	}

	@Override
	public boolean retainAll(Collection<?> c) {
		dirty = true;
		return list.retainAll(c);
	}

	@Override
	public int size() {
		int size = 0;
		for (AbstractShape s: list){
			if (s instanceof Collection<?>){
				size += ((Collection<?>) s).size();
			} else {
				size ++;
			}
		}
		return size;
	}

	@Override
	public Object[] toArray() {
		return list.toArray();
	}

	@Override
	public <T> T[] toArray(T[] a) {
		return list.toArray(a);
	}

	@Override
	public String toString(){
		return "CompoundShape with " + size() + " elements";
	}

	@Override
	public AbstractShape clone() {
		return new CompoundShape(this);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import edu.stanford.rsl.conrad.physics.PhysicalObject;

public class SimpleScene extends AbstractScene {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -4715971665107096973L;
	ArrayList<PhysicalObject> objects = new ArrayList<PhysicalObject>();
	
	public boolean add(PhysicalObject basicobj) {
		return objects.add(basicobj);
	}

	public boolean addAll(Collection<? extends PhysicalObject> scene) {
		return objects.addAll(scene);
	}
	
	public void clear() {
		objects.clear();
	}

	public boolean contains(Object o) {
		return objects.contains(o);
	}

	public boolean containsAll(Collection<?> c) {
		return objects.containsAll(c);
	}

	public boolean isEmpty() {
		return objects.isEmpty();
	}

	public Iterator<PhysicalObject> iterator() {
		return objects.iterator();
	}

	public boolean remove(Object o) {
		return objects.remove(o);
	}

	public boolean removeAll(Collection<?> c) {
		return objects.removeAll(c);
	}

	public boolean retainAll(Collection<?> c) {
		return objects.retainAll(c);
	}

	public int size() {
		return objects.size();
	}

	public Object[] toArray() {
		return objects.toArray();
	}

	public <T> T[] toArray(T[] a) {
		return objects.toArray(a);
	}

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.rendering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.physics.PhysicalObject;

public class PrioritizableScene extends AbstractScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8239645378207352898L;
	public static final boolean ADD_HIGHEST_PRIORITY = true;
	public static final boolean ADD_LOWEST_PRIORITY = false;

	private ArrayList<PhysicalObject> objects = new ArrayList<PhysicalObject>();
	private HashMap<PhysicalObject, Integer> priorityMap = new HashMap<PhysicalObject, Integer>();
	private boolean dirty = true;
	private int highest = 0;
	private int lowest = 0;

	public int getPriority(PhysicalObject o){
		//System.out.println(o.getNameString());
		return priorityMap.get(o).intValue();
	}

	/**
	 * updates the values for the priority limits
	 */
	private void updateLimits(){
		Collection<Integer> values = priorityMap.values();
		Integer [] array = new Integer[values.size()];
		if (array.length > 1){
			array = values.toArray(array);
			Arrays.sort(array);
			highest = array[array.length-1];
			lowest = array[0];
		}
		dirty = false;
	}
	
	/**
	 * Method is called after the collection changed, such that the minimum 
	 * and maximum points of the overall scene are updated. 
	 */
	protected void updateSceneLimits(){
		for (PhysicalObject e : objects)
		{
			if (this.max == null)
				this.max = new PointND(e.getShape().getMax());	
			if (this.min == null)
				this.min = new PointND(e.getShape().getMin());
			
			for (int i = 0; i < this.max.getDimension(); ++i){
				if (e.getShape().getMax().get(i) > this.max.get(i))
					this.max.set(i, e.getShape().getMax().get(i));
				if (e.getShape().getMin().get(i) < this.min.get(i))
					this.min.set(i, e.getShape().getMin().get(i));
			}
		}
	}

	public int getHighestPriority(){
		if (dirty){
			updateLimits();
		}
		return highest;
	}

	public int getLowestPriority(){
		if(dirty){
			updateLimits();
		}
		return lowest;
	}

	public boolean add(PhysicalObject e) {
		return add(e, ADD_HIGHEST_PRIORITY);
	}

	public boolean add(PhysicalObject e, boolean addMode){
		Integer priority;
		if (addMode == ADD_HIGHEST_PRIORITY){
			priority = new Integer(getHighestPriority() + 1);
		} else {
			priority = new Integer(getLowestPriority() - 1);
		}
		return add(e,priority);
	}
	
	/**
	 * This method is always called if an object is added to the scene.
	 * A call to this method updates the scene's limits (min, max).
	 * 
	 * @param e The object to add
	 * @param priority The object's priority
	 * @return true if the collection changed due to this call
	 */
	public boolean add(PhysicalObject e, int priority){
		boolean revan = objects.add(e);
		priorityMap.put(e, priority);
		dirty = true;
		updateSceneLimits();
		return revan;
	}

	public boolean addAll(Collection<? extends PhysicalObject> c) {
		boolean revan = true;
		for(PhysicalObject e:c){
			if (!add(e)){
				revan = false;
			}
		}
		return revan;
	}
	
	public boolean addAll(PrioritizableScene c) {
		boolean revan = true;
		for(PhysicalObject e:c){
			if (!add(e, c.getPriority(e))){
				revan = false;
			}
		}
		return revan;
	}

	public void clear() {
		objects.clear();
		priorityMap.clear();
		this.max = null;
		this.min = null;
	}
	
	public void clearObjectsOnly() {
		objects.clear();
		priorityMap.clear();
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
	
	public PhysicalObject getObject(int i) {
		if (i < objects.size() && i >= 0){
			return objects.get(i);
		}
		return null;
	}

	public boolean remove(Object o) {
		boolean revan = objects.remove(o);
		priorityMap.remove(o);
		dirty = true;
		updateSceneLimits();
		return revan;
	}

	public boolean removeAll(Collection<?> c) {
		boolean revan = true;
		for (Object e : c){
			if (!remove(e)){
				revan = false;
			}
		}
		updateSceneLimits();
		return revan;
	}

	public boolean retainAll(Collection<?> c) {
		boolean revan = true;
		for(Object o: this){
			if (!c.contains(o)) {
				if(!remove(o)){
					revan = false;
				}
			}
		}
		updateSceneLimits();
		return revan;
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
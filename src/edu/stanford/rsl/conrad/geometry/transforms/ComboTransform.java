package edu.stanford.rsl.conrad.geometry.transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to model a chain of transforms.
 * @author Rotimi X Ojo
 *
 */
public class ComboTransform extends Transform {

	private static final long serialVersionUID = -509282426585065975L;
	protected ArrayList<Transform> transforms = new ArrayList<Transform>();
	
	public ComboTransform(){
		
	}
	
	/**
	 * Create Combo Transform.
	 * @param t Ordered array of transforms. Transforms are applied from left to right.
	 */
	public ComboTransform(Transform...t) {
		init(t);
	}
	
	/**
	 * Initializes a Combo Transform.
	 * @param t Ordered array of transforms. Transforms are applied from left to right.
	 */
	protected void init(Transform...t){
		transforms.addAll(Arrays.asList(t));
	}
	
	/**
	 * Adds transforms to queue
	 * @param t Ordered array of transforms. Transforms are applied from left to right.
	 */
	protected void add(Transform... t){
		transforms.addAll(Arrays.asList(t));
	}
	
	
	@Override
	public PointND transform(PointND point){
		PointND p = point.clone();
		Iterator<Transform> it = transforms.iterator();		
		while(it.hasNext()){
			p = it.next().transform(p);
		}
		return p;
	}
	
	
	@Override
	public SimpleVector transform(SimpleVector dir){
		SimpleVector p = dir.clone();
		Iterator<Transform> it = transforms.iterator();		
		while(it.hasNext()){
			p = it.next().transform(p);
		}
		return p;
	}
	
	/**
	 * Inverts the transformers and order of transformation
	 */
	@Override
	public ComboTransform inverse(){
		ComboTransform inverse = new ComboTransform();		
		for(int i = transforms.size()-1; i >=0; i--){
			inverse.add(transforms.get(i).inverse());
		}
		return inverse;
	}

	@Override
	public Transform clone() {
		Transform []curr = new Transform [transforms.size()];
		for(int i = 0; i < transforms.size();i++){
			curr[i] = transforms.get(i).clone();
		}
		return new ComboTransform(curr);
	}

	@Override
	public Transform [] getData() {
		return transforms.toArray(new Transform[transforms.size()]);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.skeleton.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;


public class SkeletonCorrespondence extends HashMap< String, ArrayList<PointND> >{
	
	private static final long serialVersionUID = -3916720408279372513L;

	public void put(int[] idx, ArrayList<PointND> list){
		put(Arrays.toString(idx), list);
	}
	
	public ArrayList<PointND> get(int[] idx){
		return get(Arrays.toString(idx));
	}
	
	public void put(int idx, ArrayList<PointND> list){
		put(Integer.toString(idx), list);
	}
	
	public ArrayList<PointND> get(int idx){
		return get(Integer.toString(idx));
	}
	
}

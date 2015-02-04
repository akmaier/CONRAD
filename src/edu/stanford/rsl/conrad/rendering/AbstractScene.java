package edu.stanford.rsl.conrad.rendering;

import java.io.Serializable;
import java.util.Collection;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;


/**
 * Abstract Container for Scenes. The abstract container for all kinds of scene graphs.
 * @author akmaier
 *
 */
public abstract class AbstractScene implements Serializable, Collection<PhysicalObject> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 66628416515634675L;
	private Material background = MaterialsDB.getMaterialWithName("vacuum");
	private String name;
	protected PointND min;
	protected PointND max;
	
	
	public void setBackground(Material backgroundMaterial){
		background = backgroundMaterial;
	}
	
	public Material getBackgroundMaterial(){
		return background;
	}
	
	public void setName(String name){
		this.name = name;
	}
	
	public String getName(){
		return name;
	}

	/**
	 * @return the min
	 */
	public PointND getMin() {
		return min;
	}

	/**
	 * @param min the min to set
	 */
	public void setMin(PointND min) {
		this.min = min;
	}

	/**
	 * @return the max
	 */
	public PointND getMax() {
		return max;
	}

	/**
	 * @param max the max to set
	 */
	public void setMax(PointND max) {
		this.max = max;
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
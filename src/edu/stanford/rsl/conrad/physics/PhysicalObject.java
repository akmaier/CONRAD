package edu.stanford.rsl.conrad.physics;

import java.io.Serializable;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.physics.materials.Material;

public class PhysicalObject implements Serializable{
	
	private static final long serialVersionUID = -5731264279770114544L;
	
	protected String nameString;
	protected Material material;
	protected AbstractShape shape;
	protected double totalAbsorbedEnergy = 0;
	PhysicalObject parent = null;
		
	public PhysicalObject() {
		this.nameString = null;
		this.material = null;
		this.shape = null;
	}
	
	public PhysicalObject(PhysicalObject o) {
		this.nameString = o.nameString;
		this.material = o.material;
		this.shape = o.shape;
	}
	/**
	 * @return the nameString
	 */
	public String getNameString() {
		return nameString;
	}
	/**
	 * @param nameString the nameString to set
	 */
	public void setNameString(String nameString) {
		this.nameString = nameString;
	}
	/**
	 * @return the material
	 */
	public Material getMaterial() {
		return material;
	}
	/**
	 * @param material the material to set
	 */
	public void setMaterial(Material material) {
		this.material = material;
	}
	/**
	 * @return the shape
	 */
	public AbstractShape getShape() {
		return shape;
	}
	/**
	 * @param shape the shape to set
	 */
	public void setShape(AbstractShape shape) {
		this.shape = shape;
	}

	public ArrayList<PointND> intersect(AbstractCurve curve){
		return shape.intersect(curve);
	}
	
	public ArrayList<PointND> intersectWithHitOrientation(AbstractCurve curve){
		return shape.intersectWithHitOrientation(curve);
	}
	
	public void applyTransform(Transform t){
		shape.applyTransform(t);
	}
	
	/**
	 * absorb the energy
	 * @param location in world coordinates
	 * @param energy [eV]
	 */
	public void absorbPhoton(PointND location, double energy){
		totalAbsorbedEnergy += energy;
	}

	/**
	 * @return the totalAbsorbedEnergy in [eV]
	 */
	public double getTotalAbsorbedEnergy() {
		return totalAbsorbedEnergy;
	}

	/**
	 * @param totalAbsorbedEnergy the totalAbsorbedEnergy to set
	 */
	public void setTotalAbsorbedEnergy(double totalAbsorbedEnergy) {
		this.totalAbsorbedEnergy = totalAbsorbedEnergy;
	}

	/**
	 * @return the parent
	 */
	public PhysicalObject getParent() {
		return parent;
	}

	/**
	 * @param parent the parent to set
	 */
	public void setParent(PhysicalObject parent) {
		this.parent = parent;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
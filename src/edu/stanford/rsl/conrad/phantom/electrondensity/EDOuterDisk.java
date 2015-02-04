package edu.stanford.rsl.conrad.phantom.electrondensity;

import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.QuadricSurface;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;

/**
 * <p>Class to model the outter disk of CRIS Electron Density Phantom (Model 062)<br/>
 * Default material is plastic water.
 * @author Rotimi X Ojo
 */
public class EDOuterDisk extends PrioritizableScene{

	private static final long serialVersionUID = -6136623274828882139L;
	private QuadricDisk disk;
	private double ext_dx = 330/2, ext_dy =140,int_dx = 90, int_dy = 90, dz = 50/2;
	private double insertsDistanceFromOrigin = 115.3;
	private double insert6DistanceFromOrigin = 110;

	
	public EDOuterDisk(){
		init();
	}
	
	public EDOuterDisk(double ext_dx,double ext_dy,double int_dx, double int_dy, double dz){
		this.ext_dx = ext_dx;
		this.ext_dy = ext_dy;
		this.int_dx = int_dx;
		this.int_dy = int_dy;
		this.dz = dz;
		init();
	}
	
	private void init(){
		disk = new QuadricDisk(ext_dx, ext_dy,dz);
		disk.setMaterial(MaterialsDB.getMaterial("H2O"));
		HalfSpaceBoundingCondition cond = new HalfSpaceBoundingCondition(new Plane3D(new SimpleVector(0,1,0), -ext_dy+10));
		disk.addBoundingCondition(cond);	
		add(disk,getLowestPriority());
		add(getBackgroundDisk(new Cylinder(int_dx, int_dy, dz)),getLowestPriority() + 1);
	}

	/**
	 * Set disk material. Default material is plastic water.
	 * @param material
	 */
	public void setMaterial(Material material){
		disk.setMaterial(material);
	}

	
	private PhysicalObject getBackgroundDisk(QuadricSurface cyl){
		PhysicalObject airDisk = new PhysicalObject();
		airDisk.setShape(cyl);
		airDisk.setMaterial(getBackgroundMaterial());
		return airDisk;
	}
	
	public void addInsert(Insert ins, int index){	
		if(index == 6){
			ins.setLocation(index, insert6DistanceFromOrigin);
		}else{
			ins.setLocation(index, insertsDistanceFromOrigin);
		}		
		addAll(ins);
	}
	
}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.electrondensity;

import edu.stanford.rsl.conrad.geometry.bounds.AbstractBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.physics.PhysicalObject;

/**
 * Class to model an hollow quadric object
 * @author Rotimi X Ojo
 */
public class QuadricDisk  extends PhysicalObject{
	
	public static final long serialVersionUID = 1655681292265501419L;
	
	private Cylinder ellipt_cyl;

	public QuadricDisk(){
		
	}
	public QuadricDisk(double ext_dx, double ext_dy,  double dz) {
		init(ext_dx, ext_dy,  dz);
	}
	
	public void init(double ext_dx, double ext_dy, double dz) {
		ellipt_cyl = new Cylinder(ext_dx, ext_dy, dz);
		shape = ellipt_cyl;
		nameString = "Quadric Disk";
	}
	

	/**
	 * Add additional bounding conditions to disk.
	 * @param cond
	 */
	public void addBoundingCondition(AbstractBoundingCondition cond){
		ellipt_cyl.addBoundingCondition(cond);
	}
	
	public void applyTransform(AffineTransform affineTransform) {
		ellipt_cyl.applyTransform(affineTransform);		
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
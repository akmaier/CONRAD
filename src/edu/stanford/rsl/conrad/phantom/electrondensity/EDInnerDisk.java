package edu.stanford.rsl.conrad.phantom.electrondensity;

import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;


/**
 * Class to model the inner disk of CRIS Electron Density Phantom (Model 062)
 * Default material is plastic water.
 * @author Rotimi X Ojo
 */
public class EDInnerDisk extends PrioritizableScene{

	private static final long serialVersionUID = -4179587831440511881L;

	private QuadricDisk disk;

	private double dx = 90, dy = 90, dz = 50/2;
	private double insertsDistanceFromOrigin = 60;
	
	public EDInnerDisk(){
		init();
	}
	
	public EDInnerDisk(double dx,double dy,double dz){
		this.dx = dx;
		this.dy = dy;
		this.dz = dz;
		init();
	}
	
	private void init(){
		disk = new QuadricDisk(dx, dy, dz);
		disk.setMaterial(MaterialsDB.getMaterial("H2O"));
		add(disk,1);
	}
	
	/**
	 * Set disk material. Default material is plastic water.
	 * @param material
	 */
	public void setMaterial(Material material){
		disk.setMaterial(material);
	}
	
	public void addInsert(Insert ins, int index){
		double loc = insertsDistanceFromOrigin;
		if(index == 8){
			loc = 0;
		}

		ins.setLocation(index,loc);
		addAll(ins);
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
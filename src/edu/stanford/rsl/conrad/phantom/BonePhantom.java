package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

public class BonePhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8677909457268633175L;


	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return "Qiao & Meng Phantom";
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return "Qiao & Meng Phantom";
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "Bone Inset Phantom";
	}
	
	
	public BonePhantom(){
		PhysicalObject poObject = new PhysicalObject();
		poObject.setMaterial(MaterialsDB.getMaterial("water"));
		Cylinder clyinder = new Cylinder(120,120,100);
		poObject.setShape(clyinder);
		add(poObject);
		
		poObject = new PhysicalObject();
		poObject.setMaterial(MaterialsDB.getMaterial("bone"));
		clyinder = new Cylinder(20,20,100);
		Translation translation = new Translation(-80,0,0);
		clyinder.applyTransform(translation);
		poObject.setShape(clyinder);
		add(poObject);
		
		poObject = new PhysicalObject();
		poObject.setMaterial(MaterialsDB.getMaterial("bone"));
		clyinder = new Cylinder(20,20,100);
		translation = new Translation(0, 80,0);
		clyinder.applyTransform(translation);
		poObject.setShape(clyinder);
		add(poObject);
		
		poObject = new PhysicalObject();
		poObject.setMaterial(MaterialsDB.getMaterial("bone"));
		clyinder = new Cylinder(20,20,100);
		translation = new Translation(0, -80,0);
		clyinder.applyTransform(translation);
		poObject.setShape(clyinder);
		add(poObject);
		
		poObject = new PhysicalObject();
		poObject.setMaterial(MaterialsDB.getMaterial("bone"));
		clyinder = new Cylinder(20,20,100);
		translation = new Translation(80,0,0);
		clyinder.applyTransform(translation);
		poObject.setShape(clyinder);
		add(poObject);
		
	}
	

}

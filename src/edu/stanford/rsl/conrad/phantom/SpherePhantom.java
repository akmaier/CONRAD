package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Phantom to test forward and back projection resolution.
 * 
 * Creates a several small spheres of high contrast at several locations in the scene. (Diameter 0.1 mm)
 * The material is set to Plexiglass (density of 1.95).
 * 
 * @author Andreas Maier
 *
 */
public class SpherePhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5781085850910993618L;

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public String getName() {
		return "Sphere Phantom";
	}

	public SpherePhantom() {
		// Body of the phantom.
		double radius = 0.1;
		PointND center = new PointND(0,0,0);
		
		// center bead
		Sphere sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
		// left
		center = new PointND(100,0,0);
		sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
		// right
		center = new PointND(-100,0,0);
		sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
		// top
		center = new PointND(0, -100,0);
		sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
		// bottom
		center = new PointND(0, 100,0);
		sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
		// down
		center = new PointND(0, 0, -100);
		sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
		// up
		center = new PointND(0, 0, 100);
		sp = new Sphere(radius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
		
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
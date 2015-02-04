package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Phantom to test forward and back projection resolution.
 * 
 * Creates seven small beads of high contrast at several locations, encased in a water cylinder.
 * The material is set to Plexiglass (density of 1.95).
 * 
 * @author Martin Berger
 *
 */
public class SevenBeadPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6633919775510507739L;

	double beadRadius = 0.5;
	double beadXYdist = 0.8;
	double beadZdist = 0.8;
	double cylRadius = 100;
	double cylHeight = 200;


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
		return "Seven Bead Phantom";
	}

	@Override
	public void configure() {
		try {
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			cylRadius = UserUtil.queryDouble("Radius of the cylinder", cylRadius);
			cylHeight = UserUtil.queryDouble("Height of the cylinder", cylHeight);
			beadXYdist = UserUtil.queryDouble("Bead x/y position as a fraction of the cylinder radius", beadXYdist);
			beadZdist = UserUtil.queryDouble("Bead z position as a fraction of the cylinder height", beadZdist);
			buildPhantom();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	protected void buildPhantom(){

		Cylinder s = new Cylinder(cylRadius, cylRadius, cylHeight);
		s.setName("Water Cylinder");
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(s);
		add(po);
		
		double beadPosXY = cylRadius * beadXYdist;
		double beadPosZ = (cylHeight / 2.0) * beadZdist;
		
		// Body of the phantom.
		PointND center = new PointND(0,0,0);

		// center bead
		Sphere sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);

		// left
		center = new PointND(beadPosXY,0,0);
		sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);

		// right
		center = new PointND(-beadPosXY,0,0);
		sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);

		// top
		center = new PointND(0, -beadPosXY,0);
		sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);

		// bottom
		center = new PointND(0, beadPosXY,0);
		sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);

		// down
		center = new PointND(0, 0, -beadPosZ);
		sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);

		// up
		center = new PointND(0, 0, beadPosZ);
		sp = new Sphere(beadRadius, center);
		sp.setName("High Constrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(sp);
		add(po);
	}


	public SevenBeadPhantom() {


	}
}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
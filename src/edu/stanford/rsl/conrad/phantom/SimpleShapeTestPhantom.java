package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cone;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Ellipsoid;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SimpleSurface;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Phantom to test the projection and rendering of simple shapes
 * 
 * @author Martin Berger
 *
 */
public class SimpleShapeTestPhantom extends AnalyticPhantom {



	/**
	 * 
	 */
	private static final long serialVersionUID = 2974017903994365873L;

	public SimpleShapeTestPhantom() {
	}
	
	public SimpleShapeTestPhantom(SimpleShapeTestPhantom phant){
		buildPhantom();
		setConfigured(true);
	}

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
		return "Simple Shape Test Phantom";
	}

	@Override
	public void configure() {
		try {
			buildPhantom();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	protected void buildPhantom(){
		PhysicalObject po=null;

		Cylinder s = new Cylinder(20, 20, 40);
		Translation tr = new Translation(-40, 0, 40);
		s.applyTransform(tr);
		s.setName("Water Cylinder");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(s);
		add(po);
		
		// Body of the phantom.
		PointND center = new PointND(0,0,0);

		// water sphere
		SimpleSurface sp = new Sphere(20, center);
		tr = new Translation(40, 0, 40);
		sp.applyTransform(tr);
		sp.setName("Water Sphere");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(sp);
		add(po);

		// water box
		Box b = new Box(30,30,30);
		tr = new Translation(25, -15, -55); 
		b.applyTransform(tr);
		b.setName("Water Box");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(b);
		add(po);

		// water cone
		Cone co = new Cone(20,20,40);
		tr = new Translation(-40, 0, -20); 
		co.applyTransform(tr);
		co.setName("Water Cone");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(co);
		add(po);

		// water ellipsoid
		Ellipsoid e = new Ellipsoid(10, 15, 20);
		e.setName("Water Ellipsoid");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(e);
		add(po);
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
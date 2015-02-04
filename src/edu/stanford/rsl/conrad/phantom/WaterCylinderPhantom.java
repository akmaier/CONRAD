package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Phantom to test forward and back projection.
 * 
 * Creates a water cylinder phantom.
 * 
 * @author Martin Berger
 *
 */
public class WaterCylinderPhantom extends AnalyticPhantom {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2123220619757423791L;
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
		return "Water Cylinder Phantom";
	}

	@Override
	public void configure() {
		try {
			cylRadius = UserUtil.queryDouble("Radius of the cylinder", cylRadius);
			cylHeight = UserUtil.queryDouble("Height of the cylinder", cylHeight);
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
		po.setMaterial(MaterialsDB.getMaterialWithName("Water"));
		po.setShape(s);
		add(po);
		
	}


	public WaterCylinderPhantom() {


	}
}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
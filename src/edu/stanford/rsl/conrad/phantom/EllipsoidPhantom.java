package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Ellipsoid;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Phantom to test forward and back projection.
 * 
 * Creates a simple ellipsoid centered around the origin.
 * The material is set to Plexiglass (density of 1.95).
 * 
 *
 */
public class EllipsoidPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5997096918304998224L;


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
		return "Ellipsoid Phantom";
	}

	public EllipsoidPhantom() {
		// Body of the phantom.
		Ellipsoid theEllipsoid = new Ellipsoid(80, 80, 60);
		theEllipsoid.applyTransform(new Translation(new SimpleVector(0,0,0)));
		
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(theEllipsoid);
		
		add(po);
	}

}
/*
 * Copyright (C) 2010-2014 Zijia Guo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
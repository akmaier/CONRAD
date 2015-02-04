package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Phantom to test forward and back projection.
 * 
 * Creates a simple cube centered around the origin with a physical size of 120 mm.
 * The material is set to Plexiglass (density of 1.95).
 * 
 * @author Chris Schwemmer
 *
 */
public class BoxPhantom extends AnalyticPhantom {

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
		return "Box Phantom";
	}

	public BoxPhantom() {
		// Body of the phantom.
		Box theBox = new Box(120, 120, 120);
		theBox.applyTransform(new Translation(new SimpleVector(-60, -60, -60)));
		
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(theBox);
		
		add(po);
	}

}
/*
 * Copyright (C) 2010-2014 Chris Schwemmer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2014 Zijia Guo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cone;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Phantom to test forward and back projection.
 * 
 * Creates a simple cone with a tip at the origin.
 * The material is set to Plexiglass (density of 1.95).
 * 
 *
 */
public class ConePhantom extends AnalyticPhantom {


	/**
	 * 
	 */
	private static final long serialVersionUID = -6698697230296518461L;

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
		return "Cone Phantom";
	}

	public ConePhantom() {
		// Body of the phantom.
		Cone theCone = new Cone(120, 120, 120);
		Translation zTranslation = new Translation(0, 0, 40);
		theCone.applyTransform(zTranslation);
		
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(theCone);
		
		add(po);
		
		updateSceneLimits();
	}

}

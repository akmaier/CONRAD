/*
 * Copyright (C) 2010-2014 Zijia Guo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Pyramid;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Phantom to test forward and back projection.
 * 
 * Creates a simple pyramid with a tip at the origin.
 * The material is set to Plexiglass (density of 1.95).
 */
public class PyramidPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5599644341716177203L;

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
		return "Pyramid Shape Phantom";
	}

	public PyramidPhantom() {
		// Body of the phantom.
		Pyramid thePyramid = new Pyramid(30, 30, 80);
		thePyramid.applyTransform(new Translation(new SimpleVector(0,0,-60)));
		
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
		po.setShape(thePyramid);
		
		add(po);
	}

}

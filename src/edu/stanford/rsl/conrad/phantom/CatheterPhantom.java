package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * Phantom to test Subtract-and-Shift (SaS). 
 * 
 * @author Chris Schwemmer
 *
 */
public class CatheterPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -144401635934271936L;

	@Override
	public String getBibtexCitation() {
		return "To be published";
	}

	@Override
	public String getMedlineCitation() {
		return "To be published";
	}

	@Override
	public String getName() {
		return "Catheter Phantom";
	}

	public CatheterPhantom() {
		// Parameters
		double mainRadius = 80.0;
		double mainLength = 120.0;
		double centralRadius = 20.0;
		double centralLength = 20.0;
		double sat1Radius = 8.0;
		double sat2Radius = 4.0;
		double satLength = 70.0;
		boolean doCentre = true;
		
		// Body of the phantom
		Cylinder mainCylinder = new Cylinder(mainRadius, mainRadius, mainLength);
		
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Heart")); // D = 1.05
		po.setShape(mainCylinder);
		
		add(po);
		
		// Central, high-density cylinder
		if (doCentre) {
			Cylinder centralCylinder = new Cylinder(centralRadius, centralRadius, centralLength);
		
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName("LYSO")); // D = 5.37
			po.setShape(centralCylinder);
		
			add(po);
		}
		
		int prior = getHighestPriority() + 1;
		
		// Surrounding cylinders
		for (int i = 0; i < 8; ++i) {
			Cylinder aCylinder = new Cylinder(((i % 2) == 0) ? sat1Radius : sat2Radius,
											  ((i % 2) == 0) ? sat1Radius : sat2Radius,
											  satLength);
			
			double axisStep = (mainRadius - centralRadius) / 2.0 + centralRadius;
			double diagStep = axisStep / Math.sqrt(2);

			double dX = 0, dY = 0;
			switch (i) {
				case 0:
					dY = -axisStep;
					break;
				case 1:
					dX = diagStep;
					dY = -diagStep;
					break;
				case 2:
					dX = axisStep;
					break;
				case 3:
					dX = diagStep;
					dY = diagStep;
					break;
				case 4:
					dY = axisStep;
					break;
				case 5:
					dX = -diagStep;
					dY = diagStep;
					break;
				case 6:
					dX = -axisStep;
					break;
				case 7:
					dX = -diagStep;
					dY = -diagStep;
					break;
			}
			
			aCylinder.applyTransform(new Translation(new SimpleVector(dX, dY, 0)));
			
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName("Bone")); // D = 1.92
			po.setShape(aCylinder);
			
			add(po, prior);
		}
	}
}
/*
 * Copyright (C) 2010-2014 Chris Schwemmer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
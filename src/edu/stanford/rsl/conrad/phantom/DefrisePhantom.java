package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Creates a phantom that is scaled to the field of view of the trajectory with five disks in it.
 * With this phantom we can investigate the effect of increasing cone angle.
 * 
 * @author akmaier
 *
 */
public class DefrisePhantom extends AnalyticPhantom {


	/**
	 * 
	 */
	private static final long serialVersionUID = 6331427750379955185L;

	@Override
	public String getBibtexCitation() {
		return "@book{Zeng10-MIR,\n" +
				"  author={Zeng GL},\n" +
				"  title={Medical Image Reconstruction: A Conceptual Tutorial},\n" +
				"  publisher={Springer},\n" +
				"  location={Berlin, Heidelberg},\n" +
				"  year={2010}\n" +
				"}";
	}

	@Override
	public String getMedlineCitation() {
		return "Zeng GL. Medical Image Reconstruction: A Conceptual Tutorial. Springer Berlin/Heidelberg, 2010.";
	}

	@Override
	public String getName() {
		return "Defrise Cylinder Phantom";
	}

	public DefrisePhantom(){

		Trajectory trajectory = Configuration.getGlobalConfiguration().getGeometry();
		
		double sourceAxisDistance = trajectory.getSourceToAxisDistance();
		double sourceDetectorDistance = trajectory.getSourceToDetectorDistance();
		double detectorYAxis = trajectory.getDetectorHeight() * trajectory.getPixelDimensionY();
		double detectorXAxis = trajectory.getDetectorWidth() * trajectory.getPixelDimensionX();
		
		double fovRadius = detectorXAxis * sourceAxisDistance / sourceDetectorDistance / 2 * 0.95;
		double diskRadius = 4.0/5.0 * fovRadius;
		double fovHeight = detectorYAxis * sourceAxisDistance / sourceDetectorDistance;
		double diskHeight = fovHeight/15.0;
		double diskSpacing = fovHeight/15.0;
		int numDisks = 5;
		double diskStart =-(fovHeight / 2.0) + ((fovHeight - (numDisks*diskHeight+diskSpacing*(numDisks-1)))/2.0) + diskHeight/2.0;

		// Water Body
		Cylinder cyl = new Cylinder(fovRadius,fovRadius,fovHeight);
		cyl.setName("Water-like body of the phantom");
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.0
		po.setShape(cyl);
		add(po);

		// Disk Inserts
		for (int i=0; i < numDisks; i++){
			cyl = new Cylinder(diskRadius,diskRadius,diskHeight);
			cyl.setName("Disk");
			cyl.applyTransform(new Translation(new SimpleVector(0, 0, diskStart + (diskSpacing+diskHeight)*i)));
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass")); // D = 1.95
			po.setShape(cyl);
			add(po);
		}


	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
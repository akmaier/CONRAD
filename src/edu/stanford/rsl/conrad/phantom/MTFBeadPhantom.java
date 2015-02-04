package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * MTF phantom defined by Lars Wigstroem. The phantom is a simple elipsoid torso with two high contrast bone-like objects in it. Furthermore, there are three high-beads in the phantom that can be exploited to measure the MTF. 
 * 
 * @author akmaier
 *
 */
public class MTFBeadPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3843719205362984060L;

	@Override
	public String getName(){
		return "MTF Bead Phantom";
	}

	@Override
	public String getMedlineCitation(){
		return "Maier A, Wigström L, Hofmann H, Hornegger J, Zhu L, Strobel N, Fahrig R. Three-dimensional anisotropic adaptive filtering of projection data for noise reduction in cone beam CT. Medical Physics 38(11):5896-5909, 2011.";
	}

	@Override
	public String getBibtexCitation(){
		return "@article{Maier11-TAA,\n" +
		"  number={11},\n" +
		"  author={Andreas Maier and Lars Wigstr{\"o}m and Hannes Hofmann and Joachim Hornegger and Lei Zhu and Norbert Strobel and Rebecca Fahrig},\n" +
		"  url={http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2011/Maier11-TAA.pdf},\n" +
		"  doi={10.1118/1.3633901},\n" +
		"  journal={Medical Physics},\n" +
		"  volume={38},\n" +
		"  title={{Three-dimensional anisotropic adaptive filtering of projection data for noise reduction in cone beam CT}},\n" +
		"  year={2011},\n" +
		"  pages={5896--5909}\n" +
		"}";
	}

	/**
	 * Construtor which creates the phnatom. Internally the simple geometric objects are created and converted into physical objects by adding material to the geometry. 
	 */
	public MTFBeadPhantom () {
		// Body of the phantom.
		Cylinder mainCylinder = new Cylinder(120, 60, 300);
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithFormula("H2O")); //D = 1
		po.setShape(mainCylinder);
		add(po);
		// high contrast bone-like structures
		Cylinder left = new Cylinder(35,20,300);
		left.applyTransform(new Translation(new SimpleVector(-80,0,0)));
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Glass"));//D = 2.5
		po.setShape(left);
		add(po);
		Cylinder right = new Cylinder(35,20,300);
		right.applyTransform(new Translation(new SimpleVector(80,0,0)));
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Glass"));//D = 2.5
		po.setShape(right);
		add(po);
		// Small spheres to measure MTF.
		Sphere bead1 = new Sphere(0.75);
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterial("Al"));//D = 2.99
		po.setShape(bead1);
		add(po);
		Sphere bead2 = new Sphere(0.75);
		bead2.applyTransform(new Translation(new SimpleVector(0,40,0)));
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterial("Al"));
		po.setShape(bead2);
		add(po);
		Sphere bead3 = new Sphere(0.75);
		bead3.applyTransform(new Translation(new SimpleVector(0,-40,0)));
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterial("Al"));
		po.setShape(bead3);
		add(po);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
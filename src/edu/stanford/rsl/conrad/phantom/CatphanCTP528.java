package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
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
public class CatphanCTP528 extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3843719205362984060L;

	@Override
	public String getName(){
		return "Catphan Module CTP 528";
	}
	
	@Override
	public String getMedlineCitation(){
		return "David J. Goodenough. Catphan® 500 and 600 Manual. The Phantom Laboratory. 2009.";
	}
	
	@Override
	public String getBibtexCitation(){
		return "@book{Goodenough09-C56,\n" +
				"  author={Goodenough DJ},\n" +
				"  title={Catphan® 500 and 600 Manual},\n" +
				"  publisher={The Phantom Laboratory},\n" +
				"  year={2009}" +
				"}";
	}
	
	public static CompoundShape generateLP(double gapSize){
		int blocks = 5;
		if (gapSize >= 1) blocks = 4;
		if (gapSize >= 2.5) blocks = 3;
		if (gapSize >= 5) blocks = 2;
		double width = blocks * gapSize + (blocks-1) * gapSize;
		double height = 5;
		double centerX = width / 2;
		double centerY = height / 2;
		CompoundShape lp = new CompoundShape();
		for (int i = 0; i < blocks; i++){
			Box block = new Box(gapSize, height, height);
			block.applyTransform(new Translation((-centerX)+(i*(2*gapSize)), -centerY, -centerY));
			lp.add(block);
		}
		return lp;
	}
	
	public static double lp [] = {5, 2.5, 1.67, 1.25, 1.0, 0.83, 0.71, 0.63, 0.56, 0.5, 0.45, 0.42, 0.38, 0.36, 0.33, 0.31, 0.29, 0.28, 0.26, 0.25, 0.24};
	
	/**
	 * Construtor which creates the phnatom. Internally the simple geometric objects are created and converted into physical objects by adding material to the geometry. 
	 */
	public CatphanCTP528 () {
		// Body of the phantom.
		Cylinder mainCylinder = new Cylinder(150.49/2.0, 150.49/2.0, 30);
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithFormula("H2O")); //D = 1
		po.setShape(mainCylinder);
		add(po);
		// high contrast bone-like structures
		Sphere top = new Sphere(0.28);
		top.applyTransform(new Translation(new SimpleVector(0,20,-2.5)));
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("aluminium"));//D = 2.5
		po.setShape(top);
		add(po);
		Sphere right = new Sphere(0.28);
		right.applyTransform(new Translation(new SimpleVector(0,-20,-10)));
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("aluminium"));//D = 2.5
		po.setShape(right);
		add(po);
		double radius = 45;
		double delta = 2 * Math.PI / 23;
		for (int i=0; i < lp.length; i++){
			double angle = (delta*(1+0.7*(((double)lp.length-i)/lp.length)))*(i+1);
			ScaleRotate r = new ScaleRotate(Rotations.createBasicZRotationMatrix(Math.PI/2 +angle));
			PhysicalObject lps = new PhysicalObject();
			lps.setMaterial(MaterialsDB.getMaterialWithName("aluminium"));
			AbstractShape shape = generateLP(lp[i]);
			shape.applyTransform(r);
			double x = -Math.cos(angle)*radius;
			double y = -Math.sin(angle)*radius;
			Translation t = new Translation(new SimpleVector(x, y, 0));
			shape.applyTransform(t);
			lps.setShape(shape);
			add(lps);
		}
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
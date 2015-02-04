package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Phantom to test forward and back projection resolution.
 * 
 * Creates seven small beads of high contrast at several locations at the boundary of a water cylinder.
 * In the middle of the water cylinder, there is another cylinder representing bones.
 * The material of the beads is set to Plexiglass (density of 1.95).
 * 
 * 
 * @author Martin Berger
 *
 */
public class BeadRemovalPhantom extends AnalyticPhantom {


	/**
	 * 
	 */
	private static final long serialVersionUID = 572931430014588123L;

	int noOfBeads = 8;
	double beadRadius = 3;

	double beadZRangebegin = 0.05;
	double beadZRangeend = 0.95;

	double outerCylRadius = 100;
	double innerCylRadius = 40;
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
		return "Bead Removal Test Phantom";
	}

	@Override
	public void configure() {
		try {
			noOfBeads = UserUtil.queryInt("Number Of Beads", noOfBeads);
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			outerCylRadius = UserUtil.queryDouble("Radius of the outer cylinder", outerCylRadius);
			innerCylRadius = UserUtil.queryDouble("Radius of the inner cylinder", innerCylRadius);
			cylHeight = UserUtil.queryDouble("Height of the cylinder", cylHeight);
			buildPhantom();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public double giveRandomZBeadPosition(double[] zlimits){
		double[] minMax = DoubleArrayUtil.minAndMaxOfArray(zlimits);
		double out = Math.random()*(minMax[1]-minMax[0])+minMax[0];
		return out;
	}

	public double giveRandomAngle(double[] angleLimits){
		return giveRandomZBeadPosition(angleLimits);
	}

	protected void buildPhantom(){

		Cylinder s = new Cylinder(outerCylRadius, outerCylRadius, cylHeight);
		s.setName("Water Cylinder");
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Water")); // D = 1.95
		po.setShape(s);
		add(po);


		s = new Cylinder(innerCylRadius, innerCylRadius, cylHeight);
		s.setName("Bone Cylinder");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Femur")); // D = 1.95
		po.setShape(s);
		add(po);

		s = new Cylinder(0.9*innerCylRadius, 0.9*innerCylRadius, cylHeight);
		s.setName("Bone Marrow Cylinder");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("BoneMarrow")); // D = 1.95
		po.setShape(s);
		add(po);

		
		double beadPosRadius = outerCylRadius+0.8*beadRadius;
		double alpha = 0;
		
		double zInc = 0;
		double alphaInc = 0;
		
		double zpos = cylHeight*beadZRangebegin - cylHeight/2;

		if (noOfBeads > 1){
			alphaInc = (2*Math.PI)/(noOfBeads-1); 
			zInc = (beadZRangeend-beadZRangebegin)*cylHeight/(noOfBeads-1);
		}
			
		for (int i = 0; i < noOfBeads; i++) {
			
			PointND center = new PointND(Math.cos(alpha)*beadPosRadius, Math.sin(alpha)*beadPosRadius,zpos);
			
			// the bead
			Sphere sp = new Sphere(beadRadius, center);
			sp.setName("High Constrast Bead");
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName("SS304")); // D = 1.95
			po.setShape(sp);
			add(po);
			alpha += alphaInc;
			zpos += zInc;
		}
	}


	public BeadRemovalPhantom() {


	}
}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
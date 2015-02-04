package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class TrigonometryPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7237819451464697136L;
	private double cylRadius = 100.;
	private double cylHeight = 200.;
	private double beadRadius = 2.;
	private double bigBeadRadius = 2 * beadRadius;
	private String func = "sin";
	private String matBead = "PWO";
	private String matCyl = "Plexiglass";
	private int numberOfBeads = 24;

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getName() {
		return "Trigonometry Phantom";
	}

	@Override
	public void configure() {
		try {
			cylRadius = UserUtil.queryDouble("Cylinder radius", cylRadius);
			cylHeight = UserUtil.queryDouble("Cylinder height", cylHeight);
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			numberOfBeads = UserUtil.queryInt("Number of beads", numberOfBeads);
			func = UserUtil.queryString("Evaluation function (sin,  cos, tan)",
					func);
			matBead = UserUtil.queryString("Bead material", matBead);
			matCyl = UserUtil.queryString("Cylinder material", matCyl);
			buildPhantom();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	private void buildPhantom() {
		
		Cylinder shape = new Cylinder(cylRadius, cylRadius, cylHeight);
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName(matCyl));
		po.setShape(shape);
		add(po);
		
		for (int i = 0; i < numberOfBeads; i++) {
			
			double phi = computePhi(i);
			double x = cylRadius * Math.sin(phi);
			double y = cylRadius * Math.cos(phi);
			double z = evaluate(phi);
			
			Sphere sp = new Sphere(((i + 1) % 3 == 0) ? bigBeadRadius : beadRadius, new PointND(x, y, z));
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);
			
			//sp = new Sphere(beadRadius, new PointND(x * Math.sin(Math.PI / 2), y * Math.cos(Math.PI / 2), 0.5 * z));
			//po = new PhysicalObject();
			//po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			//po.setShape(sp);
			//add(po);
		}
		
	}
	
	private double evaluate(double phi) {
		if (func.equals("sin")) {
			return Math.sin(phi) * cylHeight / 2.0;
		} else if (func.equals("cos")) {
			return Math.cos(phi) * cylHeight / 2.0;
		} else {
			return Math.tan(phi) * cylHeight / 2.0;
		}
	}
	
	private double computePhi(int i) {
		return Math.PI * 2.0 * i / numberOfBeads;
	}

}

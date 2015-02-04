package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class XNPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9215910938649860192L;
	private double beadRadius = 1.0;
	private double cylRadius = 100;
	private double cylHeight = 200;
	private int numberOfBeads = 49;
	private int N = 4;
	private String matBead = "PWO";
	private String matCyl = "Plexiglass";

	@Override
	public String getBibtexCitation() {
		return "test";
	}

	@Override
	public String getMedlineCitation() {
		return "test";
	}

	@Override
	public String getName() {
		return "XN-Phantom";
	}

	@Override
	public void configure() {
		try {
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			cylRadius = UserUtil.queryDouble("Radius of the cylinder",
					cylRadius);
			cylHeight = UserUtil.queryDouble("Height of the cylinder",
					cylHeight);
			numberOfBeads = UserUtil.queryInt("Number of beads", numberOfBeads);
			N = UserUtil.queryInt("Exponent", N);
			buildPhantom();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private void buildPhantom() {

		// create body of the phantom
		Cylinder s = new Cylinder(cylRadius, cylRadius, cylHeight);
		s.setName("Plexiglass Cylinder");
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName(matCyl));
		po.setShape(s);
		add(po);

		// compute starting parameters
		double phi = computePhi(0);
		double x = cylRadius * Math.sin(phi);
		double y = cylRadius * Math.cos(phi);
		double z = computeZ(0);

		// define center beads
		Sphere sp = new Sphere(beadRadius, new PointND(x, y, z));
		sp.setName("High Contrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
		po.setShape(sp);
		add(po);

		sp = new Sphere(beadRadius, new PointND(-x, -y, z));
		sp.setName("High Contrast Bead");
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
		po.setShape(sp);
		add(po);

		// define string beads according to thesis
		for (int i = 1; i <= numberOfBeads / 2; i++) {

			// compute parameters of substring no 1
			phi = computePhi(i);
			x = cylRadius * Math.sin(phi);
			y = cylRadius * Math.cos(phi);
			z = computeZ(i * 2.0 / numberOfBeads);
			System.out.println("z_" + i + ": " + z);

			// define beads
			sp = new Sphere(beadRadius, new PointND(x, y, z));
			sp.setName("High Contrast Bead");
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);

			sp = new Sphere(beadRadius, new PointND(-x, -y, z));
			sp.setName("High Contrast Bead");
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);

			sp = new Sphere(beadRadius, new PointND(-x, y, -z));
			sp.setName("High Contrast Bead");
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);

			sp = new Sphere(beadRadius, new PointND(x, -y, -z));
			sp.setName("High Contrast Bead");
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);
		}
		
		System.out.println("finished");
	}

	private double computePhi(int i) {
		return (Math.PI * i) / numberOfBeads;
	}

	private double computeZ(double n_i) {
		return Math.pow(n_i, N) * cylHeight / 2.0;
	}

}

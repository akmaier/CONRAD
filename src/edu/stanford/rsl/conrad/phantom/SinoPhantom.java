package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class SinoPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4301675158282413757L;
	private double cylRadius = 100.;
	private double cylHeight = 200.;
	private int numberOfBeads = 21;
	private double beadRadius = 2.;

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
		return "Sino Phantom";
	}

	@Override
	public void configure() {
		try {
			cylHeight = UserUtil.queryDouble("Cylinder height", cylHeight);
			cylRadius = UserUtil.queryDouble("Cylinder radius", cylRadius);
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			numberOfBeads = UserUtil.queryInt("Number of beads", numberOfBeads);
			buildPhantom();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	private void buildPhantom() {
		
		Cylinder shape = new Cylinder(cylRadius, cylRadius, cylHeight);
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("Plexiglass"));
		po.setShape(shape);
		add(po);
		
		for (double phi = 0.; phi < Math.PI; phi += (2 * Math.PI / numberOfBeads)) {
			
			double z = (phi * cylHeight) / (2 * Math.PI);
			double y = Math.sin(phi) * cylRadius;
			double x = Math.cos(phi) * cylRadius;
			
			Sphere sp1 = new Sphere(beadRadius, new PointND(x, y, z));
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterial("PWO"));
			po.setShape(sp1);
			add(po);
			
			Sphere sp2 = new Sphere(beadRadius, new PointND(x, -y, -z));
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterial("PWO"));
			po.setShape(sp2);
			add(po);
			
//			Sphere sp3 = new Sphere(beadRadius, new PointND(y, 0, z));
//			po = new PhysicalObject();
//			po.setMaterial(MaterialsDB.getMaterial("PWO"));
//			po.setShape(sp3);
//			add(po);
//			
//			Sphere sp4 = new Sphere(beadRadius, new PointND(-y, 0, -z));
//			po = new PhysicalObject();
//			po.setMaterial(MaterialsDB.getMaterial("PWO"));
//			po.setShape(sp4);
//			add(po);
		}
	}

}

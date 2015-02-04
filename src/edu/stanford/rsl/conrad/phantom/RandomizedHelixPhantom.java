package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.calibration.CalibrationBead;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * currently configured for testing issues
 * 
 * @author Philipp Roser
 *
 */
public class RandomizedHelixPhantom extends AbstractCalibrationPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7656948673135057853L;

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
		return "Randomized Helix Phantom";
	}

	@Override
	public void configure() {
		try {
			/*
			 * cylRadius = UserUtil.queryDouble("Cylinder radius", cylRadius);
			 * cylHeight = UserUtil.queryDouble("Cylinder height", cylHeight);
			 * beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			 * numberOfBeads = UserUtil.queryInt("Number of beads",
			 * numberOfBeads); matBead = UserUtil.queryString("Bead material",
			 * matBead); matCyl = UserUtil.queryString("Cylinder material",
			 * matCyl);
			 */

			beadX = new double[] { -4.768299081419134, -28.583261815726104,
					-33.15013444310283, -38.28462115806606,
					-16.691248963830002, -3.4641518008626044,
					24.506062699587993, 34.94577196573563, 40.74303660269249,
					40.6571136293865, 25.47881093544778, 18.907670989310382 };
			beadY = new double[] { -40.59592281091954, -29.219219171183443,
					-23.913055250306428, 14.319686022513226, 37.31176534321907,
					40.72794221785064, 32.71419441104827, 21.20279808701686,
					-3.2818582223581454, -4.214823404029449,
					-31.962412585968956, -36.23900664698179 };
			beadZ = new double[] { 18.333333333333332, 15.0,
					11.666666666666666, 8.333333333333334, 5.0,
					1.6666666666666679, -1.6666666666666679, -5.0,
					-8.333333333333332, -11.666666666666668, -15.0,
					-18.333333333333336 };

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

		shape = new Cylinder(0.8 * cylRadius, 0.8 * cylRadius, cylHeight);
		po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName("vacuum"));
		po.setShape(shape);
		add(po);

		//beadX = new double[numberOfBeads];
		//beadY = new double[numberOfBeads];
		//beadZ = new double[numberOfBeads];

		//double phi = 2.0 * Math.PI * Math.random();

		for (int i = 1; i <= numberOfBeads; i++) {

			//double rand = Math.random();
			// double phi = 2.0 * Math.PI * rand;
			//phi += (rand); // --> randomized helix (actually
			// best)
			// phi += (2.0 * Math.PI * rand); // --> bad distribution
			// double signum = rand - 0.5;
			// phi += (2.0 * Math.PI * signum); // --> overlay

			//double x = (cylRadius - 0.125) * Math.sin(phi);
			//double y = (cylRadius - 0.125) * Math.cos(phi);
			//double z = cylHeight / 2.0 - cylHeight * ((double) i - 0.5)
			//		/ numberOfBeads;
			// boolean largeBead = i % 2 == 0 ? true : false;

			//beadX[i - 1] = x;
			//beadY[i - 1] = y;
			//beadZ[i - 1] = z;

			// boolean largeBead = i % 2 == 0 ? false : true;

			Sphere sp = new Sphere(beadRadius, new PointND(beadX[i - 1],
					beadY[i - 1], beadZ[i - 1]));
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);

		}

	}

	public void setBeadCoordinates(CalibrationBead bead, int id) {
		bead.setX(beadX[id]);
		bead.setY(beadY[id]);
		bead.setZ(beadZ[id]);
	}

	public RandomizedHelixPhantom() {
		beadX = new double[] { -4.768299081419134, -28.583261815726104,
				-33.15013444310283, -38.28462115806606,
				-16.691248963830002, -3.4641518008626044,
				24.506062699587993, 34.94577196573563, 40.74303660269249,
				40.6571136293865, 25.47881093544778, 18.907670989310382 };
		beadY = new double[] { -40.59592281091954, -29.219219171183443,
				-23.913055250306428, 14.319686022513226, 37.31176534321907,
				40.72794221785064, 32.71419441104827, 21.20279808701686,
				-3.2818582223581454, -4.214823404029449,
				-31.962412585968956, -36.23900664698179 };
		beadZ = new double[] { 18.333333333333332, 15.0,
				11.666666666666666, 8.333333333333334, 5.0,
				1.6666666666666679, -1.6666666666666679, -5.0,
				-8.333333333333332, -11.666666666666668, -15.0,
				-18.333333333333336 };
	}

	public double getX(int id) {
		return beadX[id];
	}

	public double getY(int id) {
		return beadY[id];
	}

	public double getZ(int id) {
		return beadZ[id];
	}

	public int getNumberOfBeads() {
		return numberOfBeads;
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		
	}

}

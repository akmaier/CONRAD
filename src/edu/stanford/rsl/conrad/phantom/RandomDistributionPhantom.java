package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.calibration.CalibrationBead;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * 
 * currently configured for testing issues
 * 
 * @author Philipp Roser
 *
 */
public class RandomDistributionPhantom extends AbstractCalibrationPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -789819488764215415L;
	
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
		return "Random Distribution Phantom";
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
			
			beadX = new double[] { 33.463596785306244, -39.354934605273954,
					-19.149697733008153, -29.706338119346714, 4.39305803441938,
					-11.062200522021858, -40.57471779970433, 28.918966257925426,
					-24.55910340306199, -22.19410314081742, -18.590045202218406,
					40.453397715849235 };
			beadY = new double[] { -23.4723947476699, 11.043312329849718,
					-36.111697574808396, -28.076664697557085, 40.638241424872504,
					-39.349629535874655, 4.945492945537007, 28.88700426788785,
					-32.67439463919277, 34.32473467886909, 36.40296477458225,
					-5.855615957636211 };
			beadZ = new double[] { 18.333333333333332, 15.0, 11.666666666666666,
					8.333333333333334, 5.0, 1.6666666666666679, -1.6666666666666679,
					-5.0, -8.333333333333332, -11.666666666666668, -15.0,
					-18.333333333333336 };
			buildPhantom();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	private void scale() {
		
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

		// double phi = 2.0 * Math.PI * Math.random();

		for (int i = 1; i <= numberOfBeads; i++) {

			//double rand = Math.random();
			//double phi = 2.0 * Math.PI * rand;
			// --> overlay // phi += (rand); // --> randomized helix (actually
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

	public RandomDistributionPhantom() {
		beadX = new double[] { 33.463596785306244, -39.354934605273954,
				-19.149697733008153, -29.706338119346714, 4.39305803441938,
				-11.062200522021858, -40.57471779970433, 28.918966257925426,
				-24.55910340306199, -22.19410314081742, -18.590045202218406,
				40.453397715849235 };
		beadY = new double[] { -23.4723947476699, 11.043312329849718,
				-36.111697574808396, -28.076664697557085, 40.638241424872504,
				-39.349629535874655, 4.945492945537007, 28.88700426788785,
				-32.67439463919277, 34.32473467886909, 36.40296477458225,
				-5.855615957636211 };
		beadZ = new double[] { 18.333333333333332, 15.0, 11.666666666666666,
				8.333333333333334, 5.0, 1.6666666666666679, -1.6666666666666679,
				-5.0, -8.333333333333332, -11.666666666666668, -15.0,
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

	/*public static void main(String[] args) {
		RandomPhantom phantom = new RandomPhantom();
		phantom.buildPhantom();
		System.out.print("beadX = {");
		for (int i = 0; i < phantom.numberOfBeads; i++) {
			System.out.print(" " + phantom.beadX[i]);
			if (i != phantom.numberOfBeads - 1) {
				System.out.print(",");
			} else {
				System.out.println("};");
			}
		}
		System.out.print("beadY = {");
		for (int i = 0; i < phantom.numberOfBeads; i++) {
			System.out.print(" " + phantom.beadY[i]);
			if (i != phantom.numberOfBeads - 1) {
				System.out.print(",");
			} else {
				System.out.println("};");
			}
		}
		System.out.print("beadZ = {");
		for (int i = 0; i < phantom.numberOfBeads; i++) {
			System.out.print(" " + phantom.beadZ[i]);
			if (i != phantom.numberOfBeads - 1) {
				System.out.print(",");
			} else {
				System.out.println("};");
			}
		}
	}*/

}

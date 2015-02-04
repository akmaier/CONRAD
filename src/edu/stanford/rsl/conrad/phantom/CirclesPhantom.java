package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.calibration.CalibrationBead;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class CirclesPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5200372081131255988L;
	private double cylRadius = 72.;
	private double cylHeight = 206.;
	private double beadRadius = 1.5875;
	private double bigBeadRadius = 3.175;
	private String matBead = "PWO";
	private String matCyl = "Plexiglass";
	private int numberOfBeads = 4;
	private int numberOfStrings = 4;
	private double offset = 2.0 * Math.PI / (numberOfBeads * numberOfStrings);
	private double[] beadX;
	private double[] beadY;
	private double[] beadZ;

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
		return "Circles Phantom";
	}

	@Override
	public void configure() {
		try {
			cylRadius = UserUtil.queryDouble("Cylinder radius", cylRadius);
			cylHeight = UserUtil.queryDouble("Cylinder height", cylHeight);
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
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
		
		beadX = new double[numberOfBeads * numberOfStrings];
		beadY = new double[numberOfBeads * numberOfStrings];
		beadZ = new double[numberOfBeads * numberOfStrings];
		
		int counter = 0;

		for (int i = 0; i < numberOfStrings; i++) {
			
			double height = cylHeight * (1.0 - 1.0 / (numberOfStrings * 2.0));
			System.out.println("height = " + height);
			double z =  height / 2.0 - height * i / (numberOfStrings - 1);
			System.out.println("z = " + z);

			for (int j = 0; j < numberOfBeads; j++) {

				double x = (cylRadius - 0.125)
						* Math.sin(j * 2.0 * Math.PI / numberOfBeads + i * offset);
				double y = (cylRadius - 0.125)
						* Math.cos(j * 2.0 * Math.PI / numberOfBeads + i * offset);

				Sphere sp = new Sphere(bigBeadRadius, new PointND(x, y, z));
				po = new PhysicalObject();
				po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
				po.setShape(sp);
				add(po);
				
				beadX[counter] = x;
				beadY[counter] = y;
				beadZ[counter] = z;
				counter++;
			}
		}

	}
	
	public void setBeadCoordinates(CalibrationBead bead, int id){
		bead.setX(beadX[id]);
		bead.setY(beadY[id]);
		bead.setZ(beadZ[id]);
	}
	
	public CirclesPhantom() {

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

}

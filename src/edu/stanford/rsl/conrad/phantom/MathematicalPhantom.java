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
 * @author Philipp Roser
 *
 */
public class MathematicalPhantom extends AbstractCalibrationPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1192434187777290739L;
	
	/**
	 * array that stores information about bead locations, internal
	 */
	private double[] phi;
	
	/**
	 * indicates which function is wrapped around the phantom
	 */
	private String function = "";
	private final String[] FUNCTIONS = new String[] { "Sine", "Cosine", "Sinc",
			"Tangent", "Linear", "Parabolical", "Hyperbolical" };

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
		return "Mathematical Phantom";
	}

	/**
	 * method to query function
	 * @see function
	 * @param message
	 * @param messageTitle
	 * @param functions
	 * @return a single string of the input parameter functions
	 * @throws Exception
	 */
	public static String queryFunction(String message, String messageTitle,
			String[] functions) throws Exception {
		return (String) UserUtil.chooseObject(message, messageTitle, functions,
				functions[0]);
	}
	
	/**
	 * method to query all parameters that are necessary to determine the phantoms geometry
	 */
	private void query() {
		try {
			cylRadius = UserUtil.queryDouble("Cylinder radius", cylRadius);
			cylHeight = UserUtil.queryDouble("Cylinder height", cylHeight);
			beadRadius = UserUtil.queryDouble("Bead radius", beadRadius);
			numberOfBeads = UserUtil.queryInt("Number of beads", numberOfBeads);
			matBead = UserUtil.queryString("Bead material", matBead);
			matCyl = UserUtil.queryString("Cylinder material", matCyl);
			function = queryFunction("Evaluation function", "", FUNCTIONS);
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	@Override
	public void configure() {
		try {
			query();
			buildPhantom();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

	/**
	 * sinc function
	 * @param x
	 * @return sinc of x
	 */
	private double sinc(double x) {
		return x == 0 ? 1.0 : Math.sin(x) / x;
	}

	private double sin(double x) {
		return Math.sin(x);
	}

	private double cos(double x) {
		return Math.cos(x);
	}

	private double tan(double x) {
		return Math.tan(x);
	}

	/**
	 * evaluates the function in the case it is of trigonometric origin
	 * @param func
	 * @return
	 */
	private boolean evaluateTrigonometric(String func) {
		boolean ret = false;
		phi = new double[numberOfBeads];
		for (int i = 0; i < numberOfBeads; i++) {
			double shifted = (i - numberOfBeads / 2) / 2.0;

			// phi[i] = (2.0 * Math.PI / standardDeviation)
			// * Math.exp(-0.5
			// * Math.pow(((double) (shifted - mean))
			// / standardDeviation, 2));
			if (func.equals("Sine")) {
				phi[i] = 2.0 * Math.PI * sin(shifted);
				ret = true;
			} else if (func.equals("Cosine")) {
				phi[i] = 2.0 * Math.PI * cos(shifted);
				ret = true;
			} else if (func.equals("Sinc")) {
				phi[i] = 2.0 * Math.PI * sinc(shifted);
				ret = true;
			} else if (func.equals("Tangent")) {
				phi[i] = 2.0 * Math.PI * tan(shifted);
				ret = true;
			}
		}
		
		for (int i = 1; i <= numberOfBeads; i++) {

			double x = cylRadius * Math.sin(phi[i - 1]);
			double y = cylRadius * Math.cos(phi[i - 1]);
			double z = cylHeight / 2.0 - cylHeight * ((double) i - 0.5)
					/ numberOfBeads;

			beadX[i - 1] = x;
			beadY[i - 1] = y;
			beadZ[i - 1] = z;
		}
		
		return ret;
		
	}
	
	/**
	 * evaluates the function
	 */
	private void evaluate() {
		if (evaluateTrigonometric(function)) {
			return;
		} else {
			evaluatePolynomial(function);
		}
		
	}

	/**
	 * evaluates the function in the case it is of polynomial origin
	 * @param func
	 */
	private void evaluatePolynomial(String func) {
		
		int n = 1;
		if (func.equals("Linear")) {
			n = 1;
		} else if (func.equals("Parabolical")) {
			n = 2;
		} else if (func.equals("Hyperbolical")) {
			n = 3;
		}
		
		for (int i = 1; i <= numberOfBeads; i++) {

			double z = cylHeight / 2.0 - cylHeight * ((double) i - 0.5)
					/ numberOfBeads;
			double phi = Math.pow(z, 1.0 / n);
			double x = cylRadius * Math.sin(phi);
			double y = cylRadius * Math.cos(phi);

			beadX[i - 1] = x;
			beadY[i - 1] = y;
			beadZ[i - 1] = z;
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

		beadX = new double[numberOfBeads];
		beadY = new double[numberOfBeads];
		beadZ = new double[numberOfBeads];

		evaluate();

		for (int i = 1; i <= numberOfBeads; i++) {
			
			double x = beadX[i - 1];
			double y = beadY[i - 1];
			double z = beadZ[i - 1];
			System.out.println("(" + x + ", " + y + ", " + z + ")");

			Sphere sp = new Sphere(beadRadius,
					new PointND(x, y, z));
			po = new PhysicalObject();
			po.setMaterial(MaterialsDB.getMaterialWithName(matBead));
			po.setShape(sp);
			add(po);
		}

	}

	public MathematicalPhantom() {
	
	}
	
	/**
	 * initializes the bead positions for better access in the calibration
	 * @see GeometricCalibration
	 */
	public void init() {
		query();
		
		beadX = new double[numberOfBeads];
		beadY = new double[numberOfBeads];
		beadZ = new double[numberOfBeads];

		evaluate();
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

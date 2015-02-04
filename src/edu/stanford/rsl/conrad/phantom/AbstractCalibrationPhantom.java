package edu.stanford.rsl.conrad.phantom;

import java.io.BufferedWriter;
import java.io.FileWriter;

import edu.stanford.rsl.conrad.calibration.CalibrationBead;

public abstract class AbstractCalibrationPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9152956524639810711L;
	
	/**
	 * cylinder radius
	 */
	protected double cylRadius = 41.;
	
	/**
	 * cylinder height
	 */
	protected double cylHeight = 40.;
	
	/**
	 * bead radius
	 */
	protected double beadRadius = 1.5875;
	
	/**
	 * bead material
	 */
	protected String matBead = "PWO";
	
	/**
	 * cylinder material
	 */
	protected String matCyl = "Plexiglass";
	
	/**
	 * number of beads
	 */
	protected int numberOfBeads = 12;

	/**
	 * containing x coordinates
	 */
	protected double[] beadX;
	
	/**
	 * containing y coordinates
	 */
	protected double[] beadY;
	
	/**
	 * containing z coordinates
	 */
	protected double[] beadZ;

	/**
	 * method to set 3D coordinates of a CalibrationBead
	 * @param bead
	 * @param id
	 * @see CalibrationBead
	 */
	public void setBeadCoordinates(CalibrationBead bead, int id) {
		bead.setX(beadX[id]);
		bead.setY(beadY[id]);
		bead.setZ(beadZ[id]);
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

	/**
	 * method to export the geometry to .scad. Can be further processed to .stl and 3D printed.
	 */
	public void writeToOpenSCAD() {
		try {
			configure();
			String name = getName().concat(".scad");
			BufferedWriter out = new BufferedWriter(new FileWriter(name));

			System.out.println("Initiate writing...");
			
			out.write("beadX = [");
			for (int i = 0; i < numberOfBeads - 1; i++) {
				out.write(beadX[i] + ", ");
			}
			out.write(beadX[numberOfBeads - 1] + "];\n\n");

			out.write("beadY = [");
			for (int i = 0; i < numberOfBeads - 1; i++) {
				out.write(beadY[i] + ", ");
			}
			out.write(beadY[numberOfBeads - 1] + "];\n\n");

			out.write("beadZ = [");
			for (int i = 0; i < numberOfBeads - 1; i++) {
				out.write(beadZ[i] + ", ");
			}
			out.write(beadZ[numberOfBeads - 1] + "];\n\n");

			out.write("numberOfBeads = " + numberOfBeads + ";\n\n");

			out.write("difference() {\n\n");
			out.write("cylinder(h = " + cylHeight + " , r = " + cylRadius
					+ " , $fa=1, $fs=0.1, center = true);\n");
			out.write("cylinder(h = " + (cylHeight + 1) + " , r = " + 0.8
					* cylRadius + " , $fa=1, $fs=0.1, center = true);\n");
			out.write("for (i = [0 : 1 : numberOfBeads]) { \n translate([beadX[i], beadY[i], beadZ[i]]) sphere(r = "
					+ beadRadius + ", $fa=5, $fs=0.1, center = true);\n}\n}");
			
			out.close();
			System.out.println("Writing finished...");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	abstract public void init();
}

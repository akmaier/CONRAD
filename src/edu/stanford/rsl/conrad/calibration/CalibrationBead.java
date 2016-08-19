package edu.stanford.rsl.conrad.calibration;

public class CalibrationBead implements Comparable<CalibrationBead> {

	private double u = 0;
	private double v = 0;
	private double x = 0;
	private double y = 0;
	private double z = 0;
	private Property size = Property.Large;
	
	public enum Property {Large, Small};
	
	public CalibrationBead (double u, double v){
		this.u = u;
		this.v = v;
	}

	public CalibrationBead (double u, double v, double x, double y, double z){
		this.u = u;
		this.v = v;
		this.x = x;
		this.y = y;
		this.z = z;
	}
	public double getU() {
		return u;
	}

	public void setU(double u) {
		this.u = u;
	}

	public double getV() {
		return v;
	}

	public void setV(double v) {
		this.v = v;
	}

	public double getX() {
		return x;
	}

	public void setX(double x) {
		this.x = x;
	}

	public double getY() {
		return y;
	}

	public void setY(double y) {
		this.y = y;
	}

	public double getZ() {
		return z;
	}

	public void setZ(double z) {
		this.z = z;
	}

	public Property getSize() {
		return size;
	}

	public void setSize(Property size) {
		this.size = size;
	}

	public int compareTo(CalibrationBead bead) {
		if (bead.v < v){
			return 1;
		}
		if (bead.v == v){
			return 0;
		}
		return -1;
	}
	
	@Override
	public String toString(){
		return "Bead: (" + u + ", " + v + ") ("+x+", " + y+", " +z+") " + size;
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

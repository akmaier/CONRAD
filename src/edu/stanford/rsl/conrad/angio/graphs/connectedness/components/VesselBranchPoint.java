package edu.stanford.rsl.conrad.angio.graphs.connectedness.components;

import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.BranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;

/**
 * This class represents a point in a vessel branch with information about the radius at this point.
 * @author Eric Goppert
 *
 */
public class VesselBranchPoint extends BranchPoint{
	
	private double radius = 0;
	
	private double physX_Coord = 0;
	private double physY_Coord = 0;
	private double physZ_Coord = 0;
	
	
	/**
	 * Creates a representation of a point / node of a vessel branch / tree.
	 * 
	 * @param x - x-coordinate
	 * @param y - y-coordinate
	 * @param z - z-coordinate
	 * @param radius - radius at this point
	 */
	public VesselBranchPoint(int x, int y, int z, double radius) {
		super(x, y, z, false, false);
		this.radius = radius;
	}
	
	/**
	 * Creates a representation of a point / node of a vessel branch / tree.
	 * 
	 * @param p - representation of coordinates in a Point
	 * @param radius - radius at this point
	 */
	public VesselBranchPoint(Point p, double radius) {
		super(p);
		this.radius = radius;
	}
	
	/**
	 * Creates a representation of a point / node of a vessel branch / tree inclusive physical information.
	 * 
	 * @param x - x-coordinate
	 * @param y - y-coordinate
	 * @param z - z-coordinate
	 * @param physX - physical x-coordinate
	 * @param physY - physical y-coordinate
	 * @param physZ - physical z-coordinate
	 * @param radius - radius at this point
	 */
	public VesselBranchPoint(int x, int y, int z, double physX, double physY, double physZ, double radius) {
		super(x, y, z, false, false);
		this.physX_Coord = physX;
		this.physY_Coord = physY;
		this.physZ_Coord = physZ;
		this.radius = radius;
	}
	
	/**
	 * Get coordinates of the point.
	 * 
	 * @return array with coordinates
	 */
	public int[] getCoordinates() {
		return new int[] {this.x, this.y, this.z};
	}
	
	/**
	 * Get the physical coordinates of the branch point
	 * 
	 * @return array with physical coordinates
	 */
	public double[] getPhysCoordinates() {
		return new double[] {this.physX_Coord, this.physY_Coord, this.physZ_Coord};
	}
	
	/**
	 * Set radius at respective point.
	 * 
	 * @param radius - radius at this point
	 */
	public void setRadius(double radius) {
		this.radius = radius;
	}
	
	/**
	 * Get radius at respective point.
	 * 
	 * @return radius at this point
	 */
	public double getRadius() {
		return radius;
	}
	
	/**
	 * Creates string representation for output.
	 */
	public String toString() {
		return new String("(" + this.x + ", " + this.y + ", " + this.z + ") with radius: " + this.radius);
	}

}

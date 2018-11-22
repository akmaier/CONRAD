/*
 * Copyright (C) 2010-2018 Eric Goppert
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness.components;

import java.util.ArrayList;

/**
 * Representation of a vessel branch
 * @author Eric Goppert
 *
 */
public class VesselBranch extends ArrayList<VesselBranchPoint>{

	private static final long serialVersionUID = -4680476006127812499L;
	
	private double length = 0;
	private double cost = 0;
	
	/**
	 * empty constructor
	 */
	public VesselBranch() {
		super();
	}
	
	public VesselBranch(double length) {
		super();
		this.length = length;
	}
	
	/**
	 * add point to vessel branch
	 * @param point
	 */
	public void addPoint(VesselBranchPoint point) {
		this.add(point);
	}
	
	/**
	 * get the complete branch
	 * @return array list with branch points
	 */
	public ArrayList<VesselBranchPoint> getBranch() {
		return this;
	}
	
	/**
	 * get first point of the branch
	 * @return
	 */
	public VesselBranchPoint getFirst() {
		return this.get(0);
	}
	
	/**
	 * get last point of the branch
	 * @return
	 */
	public VesselBranchPoint getLast() {
		return this.get(size() - 1);
	}
	
	public void setLength(double length) {
		this.length = length;
	}
	
	public double getLength() {
		return this.length;
	}

	public double getCost() {
		return cost;
	}

	public void setCost(double cost) {
		this.cost = cost;
	}
	
}

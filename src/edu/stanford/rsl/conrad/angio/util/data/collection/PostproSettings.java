/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.collection;

import java.io.Serializable;

public class PostproSettings implements Serializable{

	private static final long serialVersionUID = 7202210567076665344L;
	
	private double maxPointDist;
	// Minimum Spanning Forest Parameters to separate components
	private int minMstCompSize;
	// Branch extraction for every component using Dijkstra
	private int dijkstraPruning;
	
	// Significance pruning
	private double minimalLength;
	private int numSignificantBranches;
	
	// smoothing
	private int kernelSize;
	// Merging of components
	private double endPointSearchRadius;
	
	public static PostproSettings getDefaultSettings(){
		PostproSettings ps = new PostproSettings();
		ps.setMaxPointDist(5.0); // in mm
		// Min. Span Forest
		ps.setMinMstCompSize(200);
		// Min. Cost Path
		ps.setDijkstraPruning(5);
		// Significance pruning
		ps.setMinimalLength(20.0d); // in mm
		ps.setNumSignificantBranches(-1);
		// smoothing
		ps.setKernelSize(7);
		// Merging
		ps.setEndPointSearchRadius(12.0); // in mm
		
		return ps;
	}
	
	
	public double getMaxPointDist() {
		return maxPointDist;
	}
	public void setMaxPointDist(double maxMstDist) {
		this.maxPointDist = maxMstDist;
	}
	public int getMinMstCompSize() {
		return minMstCompSize;
	}
	public void setMinMstCompSize(int minMstCompSize) {
		this.minMstCompSize = minMstCompSize;
	}
	public int getDijkstraPruning() {
		return dijkstraPruning;
	}
	public void setDijkstraPruning(int dijkstraPruning) {
		this.dijkstraPruning = dijkstraPruning;
	}

	public double getEndPointSearchRadius() {
		return endPointSearchRadius;
	}


	public void setEndPointSearchRadius(double endPointSearchRadius) {
		this.endPointSearchRadius = endPointSearchRadius;
	}


	public double getMinimalLength() {
		return minimalLength;
	}


	public void setMinimalLength(double minimalLength) {
		this.minimalLength = minimalLength;
	}


	public int getNumSignificantBranches() {
		return numSignificantBranches;
	}


	public void setNumSignificantBranches(int numSignificantBranches) {
		this.numSignificantBranches = numSignificantBranches;
	}


	public int getKernelSize() {
		return kernelSize;
	}


	public void setKernelSize(int kernelSize) {
		this.kernelSize = kernelSize;
	}
	
	
	 
}

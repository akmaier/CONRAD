/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.collection;

import java.io.Serializable;

public class ReconSettings implements Serializable{	
	
	private static final long serialVersionUID = 4338057755345910927L;
	// Projection matrices
	private String pmatFile;
	// graph cut params
	private int numDepthLabels;
	private double sourceDetectorDistanceCoverage;
	private double labelCenterOffset;
	
	// exhaustive reconstruction paramters
	private double maxReprojectionError; // in mm
	private double suppressionRadius; // in mm
		
	
	public static ReconSettings getDefaultSettings(){
		ReconSettings rs = new ReconSettings();
		// Projection matrices
		rs.setPmatFile(null);
		// graph cut params
		rs.setNumDepthLabels(512);
		rs.setSourceDetectorDistanceCoverage(0.2);
		rs.setLabelCenterOffset(0.55);
		// exhaustive merging settings
		rs.setMaxReprojectionError(1.5d);
		rs.setSuppressionRadius(1.5d);
	
		return rs;
	}


	public String getPmatFile() {
		return pmatFile;
	}


	public void setPmatFile(String pmatFile) {
		this.pmatFile = pmatFile;
	}


	public double getMaxReprojectionError() {
		return maxReprojectionError;
	}


	public void setMaxReprojectionError(double maxReprojectionError) {
		this.maxReprojectionError = maxReprojectionError;
	}


	public double getSuppressionRadius() {
		return suppressionRadius;
	}


	public void setSuppressionRadius(double suppressionRadius) {
		this.suppressionRadius = suppressionRadius;
	}


	public int getNumDepthLabels() {
		return numDepthLabels;
	}


	public void setNumDepthLabels(int numDepthLabels) {
		this.numDepthLabels = numDepthLabels;
	}


	public double getSourceDetectorDistanceCoverage() {
		return sourceDetectorDistanceCoverage;
	}


	public void setSourceDetectorDistanceCoverage(
			double sourceDetectorDistanceCoverage) {
		this.sourceDetectorDistanceCoverage = sourceDetectorDistanceCoverage;
	}


	public double getLabelCenterOffset() {
		return labelCenterOffset;
	}


	public void setLabelCenterOffset(double labelCenterOffset) {
		this.labelCenterOffset = labelCenterOffset;
	}


}

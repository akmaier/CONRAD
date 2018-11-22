/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.collection;

import java.io.Serializable;

import ij.gui.Roi;

public class PreproSettings implements Serializable{
	
	private static final long serialVersionUID = 4931324893660424422L;
	
	// Bilateral filtering params
	private int bilatWidth;
	private double bilatSigmaDomain;
	private double bilatSigmaPhoto;
	
	// ECG params
	private String ecgFile;
	private double ecgWidth;
	private double ecgSlope;
	private double ecgHardThreshold;
	
	// ROI to limit processing domain
	private Roi roi;
	
	// Scales for Hessian-based processing
	private double[] hessianScales;
	private double[] gammaThreshold;
	private double structurenessPercentile;
	private double[] vesselPercentile;
	private double[] centerlinePercentile;
	
	// Connected Component Parameters
	private int conCompDilation;
	private int conCompSize;
	
	// Skeletonization and MinCostPath
	private double costMapThreshold;
	private int pruningLength;
	private int numLargestComponents; // default is 1: only one artery tree contrasted. can be 2 e.g. cavarev
	private double dijkstraMaxDistance; // currently only needed to reduce the number of edges in the graph
	
	public static PreproSettings getDefaultSettings(){
		PreproSettings preSet = new PreproSettings();
		// Bilateral filter params
		preSet.setBilatWidth(7);
		preSet.setBilatSigmaDomain(2.0);
		preSet.setBilatSigmaPhoto(0.2);
		// ECG params
		preSet.setEcgFile(null);
		preSet.setEcgWidth(0.4);
		preSet.setEcgSlope(4.0);
		preSet.setEcgHardThreshold(0.96);
		// ROI
		preSet.setRoi(new Roi(0, 0, 1024, 1024));
		// Hessian scales
		preSet.setHessianScales(new double[]{0.9, 1.2, 1.4, 1.6});
		preSet.setVesselPercentile(new double[]{0.99, 0.95});
		preSet.setGammaThreshold(new double[]{0.02,0.0085}); // phantom {0.14,0.02}
		preSet.setStructurenessPercentile(0.98);
		preSet.setCenterlinePercentile(new double[]{0.95, 0.75});
		// Connected Components
		preSet.setConCompDilation(2);
		preSet.setConCompSize(1000);
		// Skeletonization and MinCostPath
		preSet.setCostMapThreshold(0.3);
		preSet.setNumLargestComponents(1);
		preSet.setPruningLength(20);
		preSet.setDijkstraMaxDistance(5);
		return preSet;
	}
	
	public double getEcgWidth() {
		return ecgWidth;
	}
	public void setEcgWidth(double ecgWidth) {
		this.ecgWidth = ecgWidth;
	}
	public double getEcgSlope() {
		return ecgSlope;
	}
	public void setEcgSlope(double ecgSlope) {
		this.ecgSlope = ecgSlope;
	}

	public Roi getRoi() {
		return roi;
	}

	public void setRoi(Roi roi) {
		this.roi = roi;
	}

	public double[] getHessianScales() {
		return hessianScales;
	}

	public void setHessianScales(double[] hessianScales) {
		this.hessianScales = hessianScales;
	}

	public String getEcgFile() {
		return ecgFile;
	}

	public void setEcgFile(String ecgFile) {
		this.ecgFile = ecgFile;
	}

	public double getEcgHardThreshold() {
		return ecgHardThreshold;
	}

	public void setEcgHardThreshold(double ecgHardThreshold) {
		this.ecgHardThreshold = ecgHardThreshold;
	}

	public int getBilatWidth() {
		return bilatWidth;
	}

	public void setBilatWidth(int bilatWidth) {
		this.bilatWidth = bilatWidth;
	}

	public double getBilatSigmaDomain() {
		return bilatSigmaDomain;
	}

	public void setBilatSigmaDomain(double bilatSigmaDomain) {
		this.bilatSigmaDomain = bilatSigmaDomain;
	}

	public double getBilatSigmaPhoto() {
		return bilatSigmaPhoto;
	}

	public void setBilatSigmaPhoto(double bilatSigmaPhoto) {
		this.bilatSigmaPhoto = bilatSigmaPhoto;
	}

	public double[] getVesselPercentile() {
		return vesselPercentile;
	}

	public void setVesselPercentile(double[] vesselPercentile) {
		this.vesselPercentile = vesselPercentile;
	}

	public double[] getCenterlinePercentile() {
		return centerlinePercentile;
	}

	public void setCenterlinePercentile(double[] centerlinePercentile) {
		this.centerlinePercentile = centerlinePercentile;
	}

	public int getConCompSize() {
		return conCompSize;
	}

	public void setConCompSize(int conCompSize) {
		this.conCompSize = conCompSize;
	}

	public int getConCompDilation() {
		return conCompDilation;
	}

	public void setConCompDilation(int conCompDilation) {
		this.conCompDilation = conCompDilation;
	}

	public double getCostMapThreshold() {
		return costMapThreshold;
	}

	public void setCostMapThreshold(double costMapThreshold) {
		this.costMapThreshold = costMapThreshold;
	}

	public int getPruningLength() {
		return pruningLength;
	}

	public void setPruningLength(int pruningLength) {
		this.pruningLength = pruningLength;
	}

	public double getDijkstraMaxDistance() {
		return dijkstraMaxDistance;
	}

	public void setDijkstraMaxDistance(double dijkstraMaxDistance) {
		this.dijkstraMaxDistance = dijkstraMaxDistance;
	}

	public int getNumLargestComponents() {
		return numLargestComponents;
	}

	public void setNumLargestComponents(int numLargestComponents) {
		this.numLargestComponents = numLargestComponents;
	}

	public double[] getGammaThreshold() {
		return gammaThreshold;
	}

	public void setGammaThreshold(double[] gammaThreshold) {
		this.gammaThreshold = gammaThreshold;
	}

	public double getStructurenessPercentile() {
		return structurenessPercentile;
	}

	public void setStructurenessPercentile(double structurenessPercentile) {
		this.structurenessPercentile = structurenessPercentile;
	}
	
	
	
}

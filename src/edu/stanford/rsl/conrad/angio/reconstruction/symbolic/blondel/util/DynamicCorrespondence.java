package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class DynamicCorrespondence {
	PointND point;
	ArrayList<PointND> correspondences;
	ArrayList<PointND> points3D;
	double[] errorList;
	double[][] connectivity;
	int[] slidesNr;
	int branchNr;
	int pointNr;
	
	public DynamicCorrespondence(){	
	}
	

	public DynamicCorrespondence(int[] slidesNr, int branchNr, int pointNr, PointND point, double[] error, double[][] connectivity){
		this.slidesNr = slidesNr;
		this.branchNr = branchNr;
		this.pointNr = pointNr;
		this.point = point;
		this.errorList = error;
		this.connectivity = connectivity;
	}
	public int[] getSlidesNr() {
		return slidesNr;
	}

	public void setSlidesNr(int[] slidesNr) {
		this.slidesNr = slidesNr;
	}
	
	public ArrayList<PointND> getPoints3D() {
		return points3D;
	}


	public void setPoints3D(ArrayList<PointND> points3d) {
		points3D = points3d;
	}


	public ArrayList<PointND> getCorrespondences() {
		return correspondences;
	}

	public void setCorrespondences(ArrayList<PointND> correspondences) {
		this.correspondences = correspondences;
	}
	public int getBranchNr() {
		return branchNr;
	}

	public void setBranchNr(int branchNr) {
		this.branchNr = branchNr;
	}

	public int getPointNr() {
		return pointNr;
	}

	public void setPointNr(int pointNr) {
		this.pointNr = pointNr;
	}	
	public PointND getPoint(){
		return point;
	}
	public double[] getErrorList(){
		return errorList;
	}
	public double[][] getConnectivity(){
		return connectivity;
	}
	public void setPoint(PointND newPoint){
		this.point = newPoint;
	}
	public void setErrorList(double[] error){
		this.errorList = error;
	}
	public void setConnectivity(double[][] connectedTo){
		this.connectivity = connectedTo;
	}
}

/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.organization;

import java.util.ArrayList;

import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;

/**
 * Class to store an angiographic acquisition using a Grid3D to store the projection images, a projection array to 
 * store the projection matrices and a double array to store the heart phase obtained using ECG signals.
 * 
 * @author Mathias Unberath
 *
 */
public class Angiogram {
	
	private Projection[] pMatrices = null;
	private Grid3D projections = null;
	private Grid3D reconCostMap = null;
	private double[] ecg = null;
	private double[] primAngles = null;
	private double[] secAngles = null;
	private ArrayList<Skeleton> skeletons = null;
	private int numProjections;

	public Angiogram(ImagePlus p, Projection[] pMat, double[] angles, double[] e){
		this.projections = ImageUtil.wrapImagePlus(p);
		this.pMatrices = pMat;
		this.ecg = e;
		this.setPrimAngles(angles);
		this.setNumProjections(pMat.length);
	}
	
	public Angiogram(ImagePlus p, Projection[] pMat, double[] primAngles, double[] secAngles, double[] e){
		this.projections = ImageUtil.wrapImagePlus(p);
		this.pMatrices = pMat;
		this.ecg = e;
		this.primAngles = primAngles;
		this.secAngles = secAngles;
		this.setNumProjections(pMat.length);
	}
	
	public Angiogram(ImagePlus p, Projection[] pMat, double[] e){
		this.projections = ImageUtil.wrapImagePlus(p);
		this.pMatrices = pMat;
		this.ecg = e;
		double[] angles = new double[ecg.length];
		for(int i = 0; i < angles.length; i++){
			angles[i] = 200.0/(angles.length-1)*i;
		}
		this.primAngles = angles;
		this.setNumProjections(pMat.length);
	}
	
	public Angiogram(Grid3D p, Projection[] pMat, double[] angles, double[] e){
		this.projections = p;
		this.pMatrices = pMat;
		this.ecg = e;
		this.setPrimAngles(angles);
		this.setNumProjections(pMat.length);
	}

	public Angiogram(Grid3D p, Projection[] pMat, double[] primAngles, double[] secAngles, double[] e){
		this.projections = p;
		this.pMatrices = pMat;
		this.ecg = e;
		this.primAngles = primAngles;
		this.secAngles = secAngles;
		this.setNumProjections(pMat.length);
	}
	
	public Angiogram(Grid3D p, Projection[] pMat, double[] e){
		this.projections = p;
		this.pMatrices = pMat;
		this.ecg = e;
		double[] angles = new double[ecg.length];
		for(int i = 0; i < angles.length; i++){
			angles[i] = 200.0/(angles.length-1)*i;
		}
		this.primAngles = angles;
		this.setNumProjections(pMat.length);
	}
	
	public Angiogram(Angiogram a){
		this.projections = (Grid3D)a.getProjections().clone();
		this.pMatrices = a.getPMatrices().clone();
		this.ecg = a.getEcg().clone();
		this.primAngles = (a.getPrimAngles()!=null)?a.getPrimAngles().clone():null;
		this.secAngles = (a.getSecondaryAngles()!=null)?a.getSecondaryAngles().clone():null;
		this.setNumProjections(a.getNumProjections());
	}
	
	public Projection[] getPMatrices() {
		return pMatrices;
	}

	public void setpMatrices(Projection[] pMatrices) {
		this.pMatrices = pMatrices;
	}

	public Grid3D getProjections() {
		return projections;
	}

	public void setProjections(Grid3D projections) {
		this.projections = projections;
	}

	public double[] getEcg() {
		return ecg;
	}

	public void setEcg(double[] ecg) {
		this.ecg = ecg;
	}

	public double[] getPrimAngles() {
		return primAngles;
	}

	public void setPrimAngles(double[] primAngles) {
		this.primAngles = primAngles;
	}
	
	public double[] getSecondaryAngles() {
		return secAngles;
	}

	public void setSecondaryAngles(double[] secAngles) {
		this.secAngles = secAngles;
	}

	public ImagePlus getProjectionsAsImP() {
		return ImageUtil.wrapGrid3D(this.projections, "Projections");
	}

	public ArrayList<Skeleton> getSkeletons() {
		return skeletons;
	}

	public void setSkeletons(ArrayList<Skeleton> skeletons) {
		this.skeletons = skeletons;
	}

	public Grid3D getReconCostMap() {
		return reconCostMap;
	}

	public void setReconCostMap(Grid3D reconCostMap) {
		this.reconCostMap = reconCostMap;
	}

	public int getNumProjections() {
		return numProjections;
	}

	public void setNumProjections(int numProjections) {
		this.numProjections = numProjections;
	}

}

package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.ECG;

import ij.ImagePlus;
import edu.stanford.rsl.conrad.geometry.Projection;

/**
 * Class to store an angioraphic acquisition using an ImagePlus to store the projection images, a projection array to 
 * store the projection matrices and a double array to store the heart phase obtained using ECG signals.
 * @author Mathias
 *
 */
public class Angiogram {
	
	private Projection[] pMatrices = null;
	private ImagePlus projections = null;
	private double[] ecg = null;
	private double[] primAngles = null;

	public Angiogram(ImagePlus p, Projection[] pMat, double[] angles, double[] e){
		this.projections = p;
		this.pMatrices = pMat;
		this.ecg = e;
		this.setPrimAngles(angles);
	}

	public Projection[] getPMatrices() {
		return pMatrices;
	}

	public void setpMatrices(Projection[] pMatrices) {
		this.pMatrices = pMatrices;
	}

	public ImagePlus getProjections() {
		return projections;
	}

	public void setProjections(ImagePlus projections) {
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
	

}

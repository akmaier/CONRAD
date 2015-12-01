package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.ECG;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.ECG.Angiogram;

public class ECGGating {
	
	private double width = 0.4; //windowWith [0,1]
	private double exp = 4.0; // exponentA >= 0
	private double refHeartPhase = 0.9; // refHeartPhase [0,1]

	public ECGGating(double width, double exp, double hp){
		assert(width <= 1 && width >= 0) : new IllegalArgumentException("Width needs to be in [0,1].");
		assert(exp >= 1) : new IllegalArgumentException("Exp needs to be >1.");
		assert(hp <= 1 && hp >= 0) : new IllegalArgumentException("Heart Phase needs to be in [0,1].");
		this.width = width;
		this.exp = exp;
		this.refHeartPhase = hp;
	}

	/**
	 * Calculates the ECG-gating based weights of projections corresponding to the ECG signal handled.
	 * @param ecg
	 * @return
	 */
	public double[] evaluate(double[] ecg){
		double[] weights = new double[ecg.length];
		for(int i = 0; i < ecg.length; i++){
			weights[i] = getWeight(ecg[i]);
		}
		return weights;
	}
	
	
	private double getWeight(double currentHeartPhase){
		double weight = 0;
		double distMeasure = getDistanceMeasure(currentHeartPhase);
		if(distMeasure > width/2){
			return 0;
		}else{
			double cos = Math.cos((distMeasure/width)*Math.PI);
			weight = Math.pow(cos, exp);
		}		
		return weight;
	}
	
	
	private double getDistanceMeasure(double currentHeartPhase){

		int[] j = new int[] {-1,0,+1};
		
		double min = Math.abs(currentHeartPhase - refHeartPhase + j[0]);
		min = Math.min((Math.abs(currentHeartPhase - refHeartPhase + j[1])),min);
		min = Math.min((Math.abs(currentHeartPhase - refHeartPhase + j[2])),min);
		
		return min;
	}
	
	public Grid3D weightProjections(Grid3D g, double[] ecg){
		double[] weights = evaluate(ecg);
		Grid3D weighted = new Grid3D(g);
		for(int k = 0; k < g.getSize()[2]; k++){
			float weight = (float)weights[k];
			for(int i = 0; i < g.getSize()[0]; i++){
				for(int j = 0; j < g.getSize()[1]; j++){
					weighted.setAtIndex(i, j, k, g.getAtIndex(i, j, k)*weight);
				}
			}	
		}
		return weighted;
	}
	
	/**
	 * Applies gating to an angiogram using the specified threshold for heart-phase weights.
	 * Angiograms consist of projection matrices, projection images and the corresponding ECG signal. 
	 * @param a
	 * @param threshold
	 * @return
	 */
	public Angiogram applyGating(Angiogram a, double threshold){
		Projection[] pMat = a.getPMatrices();
		ImageStack ims = a.getProjections().getImageStack();
		double[] primA = a.getPrimAngles();
		double[] ecg = a.getEcg();
		
		double[] weights = evaluate(ecg);
		int nAbove = 0;
		for(int i = 0; i < ecg.length; i++){
			if(weights[i] > threshold){
				nAbove++;
			}
		}
		double[] reducedEcg = new double[nAbove];
		double[] reducedPrimA = new double[nAbove];
		ImageStack reducedIms = new ImageStack(ims.getProcessor(1).getWidth(), ims.getProcessor(1).getHeight());
		Projection[] reducedPMat = new Projection[nAbove];
		int count = 0;
		for(int i = 0; i < weights.length; i++){
			if(weights[i] > threshold){
				reducedEcg[count] = ecg[i];
				reducedIms.addSlice(ims.getProcessor(i+1));
				reducedPrimA[count] = primA[i];
				reducedPMat[count] = pMat[i];
				count++;
			}
		}
		ImagePlus reducedImp = new ImagePlus();
		reducedImp.setStack(reducedIms);
		return new Angiogram(reducedImp, reducedPMat, reducedPrimA, reducedEcg);
	}
	
	/**
	 * Applies gating to an angiogram by multiplication of the projection with the corresponding weight.
	 * Angiograms consist of projection matrices, projection images and the corresponding ECG signal. 
	 * @param a
	 * @return
	 */
	public Angiogram applyGating(Angiogram a){
		Projection[] pMat = a.getPMatrices();
		ImageStack ims = a.getProjections().getImageStack();
		double[] primA = a.getPrimAngles();
		double[] ecg = a.getEcg();
		
		double[] weights = evaluate(ecg);
		ImageStack weightedIms = new ImageStack(ims.getProcessor(1).getWidth(), ims.getProcessor(1).getHeight());
		for(int i = 0; i < weights.length; i++){
			FloatProcessor current = (FloatProcessor) ims.getProcessor(i+1);
			current.multiply(weights[i]);
			weightedIms.addSlice(current);			
		}
		ImagePlus reducedImp = new ImagePlus();
		reducedImp.setStack(weightedIms);
		return new Angiogram(reducedImp, pMat, primA, ecg);
	}
	
}

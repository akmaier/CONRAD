package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.Grid;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.tutorial.phantoms.Phantom;

public class ART {
	protected int subsetSize; 
	protected int numberOfSubsets;
	private double error;
	private double epsilon;
	
	private int maxIter;
	private int iter;
	
	protected Projector projector;
	private Backprojector backprojector;
	
	protected double stepSize;

	public ART(Projector projector, Backprojector backprojector){
		this.projector = projector;
		this.backprojector =  backprojector;
	}
	
	public double calculateError(Grid2D grid, Grid2D recon) {
		double calError = 0.0;
		for (int j = 0; j < recon.getSize()[1]; j++) {
			for (int k = 0; k < recon.getSize()[0]; k++) {
				calError += Math.pow(
						recon.getAtIndex(j, k) - grid.getAtIndex(j, k), 2);
			}
		}
		calError = (Math.sqrt(calError)) / (grid.getSize()[0] * grid.getSize()[1]);
		return calError;
	}
	public void setNumberOfSubsets(int numberOfSubsets){
		this.numberOfSubsets = numberOfSubsets;
		this.subsetSize = 180 / numberOfSubsets; 

	}
	public void setIterationEnvironment(int maxIter, double epsilon){
		this.maxIter = maxIter;
		this.iter = 0;	
		this.error = 100;
		this.epsilon = epsilon;
	}
	
	public void setStepSize(Grid recon){
		this.stepSize =-1.0/(Math.sqrt(Math.pow(recon.getSize()[0],2)+Math.pow(recon.getSize()[1],2)));		
	}
	
	public double getPhanError(Phantom phan, Grid2D recon){
		return this.calculateError(phan, recon);
		
	}

	public void shortPause(){
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public NumericGrid reconstruct(NumericGrid originalSinogram, NumericGrid recon, NumericGrid imageUpdate, NumericGrid localImageUpdate, NumericGrid diff) {
		Grid2D reconOld = null;
		// long startTime = System.currentTimeMillis();		
		while (error > epsilon && iter < maxIter) {
			if (recon!= null) reconOld = (Grid2D)recon.clone();
			error = 0;

			for (int k = 0; k < numberOfSubsets; k++) {
				NumericPointwiseOperators.fill(imageUpdate, 0);

				for (int i = 0; i < subsetSize; i++) {
					int index = (i * numberOfSubsets + k) % 180;

					diff = projector.project(recon, diff, index);
					NumericPointwiseOperators.subtractBy(diff,originalSinogram.getSubGrid(index));

					localImageUpdate = backprojector.backproject(diff, localImageUpdate, index);
					NumericPointwiseOperators.addBy(imageUpdate, localImageUpdate);
//					localImageUpdate.clone().show("Reconstruction " + iter);
					setDiff(diff, i);
				}

				imageUpdate(imageUpdate);
				setStepSize(recon);
				//imageUpdate.clone().show("Reconstruction " + iter);
				
				NumericPointwiseOperators.multiplyBy(imageUpdate, (float) this.stepSize);
				NumericPointwiseOperators.addBy(recon, imageUpdate);

			}

			if (recon != null) error = calculateError(reconOld, (Grid2D) recon);
	
			iter++;

			if (iter % 10 == 0 || iter == 1) {
				recon.clone().show("Reconstruction " + iter);
				System.out.println("Iteration: " + iter);
				System.out.println("Recon: " + error);
			}
			System.out.println("Iteration: " + iter
					+ " current error: " + error
					+ " current step size: " + stepSize);
		}
//		recon.clone().show("Reconstruction " + iter  +" final result" );
//		System.out.println("Recon final error: " + error);
		
		return recon;

	}

	// This is only implemented for ARTStepSizeControl
	public void imageUpdate(Grid imageUpdate) {
		// TODO Auto-generated method stub
		
	}

	// This is only implemented for ARTStepSizeControl
	public void setDiff(Grid diff, int i) {
		// TODO Auto-generated method stub
		
	}
}

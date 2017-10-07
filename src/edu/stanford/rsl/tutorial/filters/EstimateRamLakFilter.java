/*
 * Copyright (C) 2010-2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.filters;

import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;
import ij.ImageJ;

/**
 * Demonstration to learn the RamLakConvolver including discretization from the analytic solution using the ramp filter.
 * @author maier
 *
 */
public class EstimateRamLakFilter extends RamLakKernel implements GridKernel  {


	public EstimateRamLakFilter(int size, double deltaS) {
		super(size, deltaS);
		// Initializaiton using simple abs of omega
		double step = 0.5 / (this.getNumberOfElements()/4);
		for (int i= 0; i < this.getNumberOfElements()/4;i++){
			this.buffer[i*2] = (float) (i*step);
			this.buffer[i*2+1] = 0;
		}
		for (int i= this.getNumberOfElements()/4;i < this.getNumberOfElements()/2;i++){
			this.buffer[i*2] = (float) (0.5-(((i-this.getNumberOfElements()/4)+1)*step));
			this.buffer[i*2+1] = 0; 
		}
	}
	
	/**
	 * Performs one update step on the filter using a simple gradient descent.
	 * Projections p are formed from the slice x using A.
	 * p = Ax
	 * Radon inverse A^(-1) is computed as
	 * x = A^(-1) p = A^T (A^T A)^(-1) p =
	 *   = A^T F^H R F p
	 * where A^T is the back projection, F the Fourier transform, and R a diagonal matrix with the ramp filter
	 * Now we want to estimate R by minimization of f(R), (using some identities from the matrix cookbook below...):
	 * f(R) = 0.5 || x - A^T F^H R F p ||_2^2 = 
	 *      = 0.5 (x - A^T F^H R F p)^T (x - A^T F^H R F p) =
	 *      = 0.5 (x^T - p^T F^H R^T F A) (x - A^T F^H R F p) =
	 *      = 0.5 (x^T x - p^T F^H R^T F A x - x^T A^T F^H R F p + p^T F^H R^T F A A^T F^H R F p)
	 * Computation of the gradient:
	 * f(R) / dR = 0.5 ( - F A x p^T F^H - (x^T A^T F^H)^T (F p)^T  + (F A A^T F^H)^T R (p^T F^H)^T (F p)^T + F A A^T F^H R F p p^T F^H) = 
	 *           = 0.5 ( - F A x p^T F^H - F A x p^T F^H + F A A^T F^H R F p p^T F + F A A^T F^H R F p p^T F^H) =
	 *           = F A A^T F^H R F p p^T F^H - F A x p^T F^H
	 *           = F A (A^T F^H R F p - x) p^T F^H
	 *           = F A (A^T F^H R F p - x) (F p)^T
	 *                  ^^^^Recon^^^^
	 * Algorithm for gradient is then:
	 * FFT(Project(Recon - Image)) * FFT(Sinogram)                           
	 *           
	 * Sanity Check by setting gradient to 0:
	 * f(R) / dR != 0
	 * F A A^T F^H R F p p^T F^H = F A x p^T F^H
	 * A^T F^H R F p = x 
	 * -> This is exactly the FBP reconstruction formula.
	 * @param Sino the observed sinogram
	 * @param reference correct reference on x
	 */
	public void updateFilter(ParallelProjector2D projector, ParallelBackprojector2D backproj, Grid2D sinogram, Grid2D reference, float regularization, float stepSize, int regCount){
		Grid2D currentRecon = recon(sinogram, backproj);
		NumericGridOperator op = currentRecon.getGridOperator();
		//op.removeNegative(currentRecon);
		op.subtractBy(currentRecon, reference);
		Grid2D updateSino = projector.projectRayDrivenCL(currentRecon);
		Grid1DComplex acc = new Grid1DComplex(sinogram.getSize()[0]);
		double sumReal = 0;
		double sumImag = 0;
		double scale = acc.getNumberOfElements()*sinogram.getSize()[1] *acc.getNumberOfElements()*sinogram.getSize()[1];
		for (int theta = 0; theta < sinogram.getSize()[1]; ++theta) {
			Grid1DComplex currentProj = new Grid1DComplex(sinogram.getSubGrid(theta));
			currentProj.transformForward();
			Grid1DComplex currentUpdate = new Grid1DComplex(updateSino.getSubGrid(theta));
			currentUpdate.transformForward();
			for (int i=0; i < currentUpdate.getNumberOfElements()/2; i++){
				currentUpdate.multiplyAtIndex(i, currentProj.getRealAtIndex(i), currentProj.getImagAtIndex(i));
				acc.addAtIndex(i, (float) (currentUpdate.getRealAtIndex(i) / (scale)), (float) (currentUpdate.getImagAtIndex(i)/ (scale)));
				sumReal += currentUpdate.getRealAtIndex(i)/scale;
				sumImag += currentUpdate.getImagAtIndex(i)/scale;
			}		
		}
		//double sum = op.sum(acc);
		
		double sum = Math.sqrt(sumReal*sumReal+sumImag*sumImag);
		for (int i=0; i<regCount; i++) smooth(acc,regularization);
		System.out.println(sumReal + " "+sumImag + " " +sum + " " + op.max(acc)*stepSize);
		//op.divideBy(acc, 1000000);
		//acc.show("Gradient");
		
		for (int i=0; i < acc.getNumberOfElements()/2; i++){
			this.addAtIndex(i, -acc.getRealAtIndex(i)*stepSize, -acc.getImagAtIndex(i)*stepSize*0.f);
		}
		//updateSino.show();
		
	}
	
	/**
	 * little helper tool to smooth the gradient update before application. This helps a lot with numerical stability
	 * @param grid
	 * @param weight
	 */
	public void smooth(Grid1DComplex grid, float weight){
		Grid1DComplex temp = new Grid1DComplex(grid);
		temp.setAtIndex(0, 
				(grid.getRealAtIndex(0)+grid.getRealAtIndex(1)*weight) /(1.0f + weight),
				(grid.getImagAtIndex(0)+grid.getImagAtIndex(1))*weight /(1.0f + weight));
		int j = grid.getNumberOfElements()/2-1;
		temp.setAtIndex(j, 
				(grid.getRealAtIndex(j)+grid.getRealAtIndex(j-1)*weight) /(1.0f + weight),
				(grid.getImagAtIndex(j)+grid.getImagAtIndex(j-1)*weight) /(1.0f + weight));
		for (int i=1; i < this.getNumberOfElements()/2-1; i++){
			temp.setAtIndex(i, 
					(grid.getRealAtIndex(i)+grid.getRealAtIndex(i-1)*weight+grid.getRealAtIndex(i+1)*weight) /(1.0f + 2*weight),
					(grid.getImagAtIndex(i)+grid.getImagAtIndex(i-1)*weight+grid.getImagAtIndex(i+1)*weight) /(1.0f + 2*weight));
		}
		for (int i=0; i < this.getNumberOfElements()/2; i++){
			grid.setAtIndex(i, 
					temp.getRealAtIndex(i),
					temp.getImagAtIndex(i));
		}
	}
	
	/**
	 * Courtesy function for reconstruction
	 * @param sinogram
	 * @param backproj
	 * @return
	 */
	public Grid2D recon(Grid2D sinogram, ParallelBackprojector2D backproj){	
		Grid2D filteredSinogram = new Grid2D(sinogram);
		// Filter with current solution
		for (int theta = 0; theta < sinogram.getSize()[1]; ++theta) {
			this.applyToGrid(filteredSinogram.getSubGrid(theta));
		}
		return backproj.backprojectPixelDriven(filteredSinogram);
	}

	/**
	 * Compare estimated RamLak with true one and the ramp filter to show the differences.
	 * @param args
	 */
	public static void main(String[] args) {
		// comparision of the two filters.
		
		new ImageJ();
	
		// Parameters:
		int x = 100;
		int y = 100;
		double deltaS = 1;
		int size = 200;
		int numphan = 10;
		int iterations = 20;
		float stepSize = 0.01f;
		int regCount = 1;
		float regularization = 1.0f;
		double angleStep = Math.PI/180.0;
		
		
		
		EstimateRamLakFilter estimated = new EstimateRamLakFilter(size, deltaS);
		EstimateRamLakFilter ramp = new EstimateRamLakFilter(size, deltaS);
		estimated.show("Initial Ramp");
		RamLakKernel ramLak = new RamLakKernel(size, deltaS);
		ramLak.show("RamLak");
		NumericGridOperator op = estimated.getGridOperator();
		//op.subtractBy(estimated, ramLak);
		//estimated.show();
		
		

		// Create a phantom
		
		Phantom[] phan = new Phantom[numphan];
		Grid2D[] sinogram = new Grid2D[numphan];
		ParallelProjector2D projector = new ParallelProjector2D(Math.PI*2, angleStep, size*deltaS, deltaS);
		ParallelBackprojector2D backproj = new ParallelBackprojector2D(x, y, 1, 1);
		
		for (int i=0; i<numphan; i++){
			phan[i] = new UniformCircleGrid2D(x, y,((double)(i+numphan))/(numphan*4));
			//phan[i].show("The Phantom");
			sinogram[i] = projector.projectRayDrivenCL(phan[i]);
			//sinogram[i].show("The Projection");	
		}
		//phan = new UniformCircleGrid2D(x, y);
		//phan = new MickeyMouseGrid2D(x, y);
		
		
		// Project forward parallel
		//sinogram.show("The Sinogram");
		Grid2D filteredSinogram = new Grid2D(sinogram[numphan-1]);
		Grid2D filteredSinogramRamp = new Grid2D(sinogram[numphan-1]);
		Grid2D filteredSinogramEst = new Grid2D(sinogram[numphan-1]);
		for (int i=0; i <iterations; i++){
			for (int p=0; p<numphan; p++){
				estimated.updateFilter(projector, backproj, sinogram[p], phan[p], regularization, stepSize, regCount);
			}
			//estimated.show("Iteration " + i);
		}
		
		// Filter with RamLak
		for (int theta = 0; theta < sinogram[numphan-1].getSize()[1]; ++theta) {
			ramLak.applyToGrid(filteredSinogram.getSubGrid(theta));
			ramp.applyToGrid(filteredSinogramRamp.getSubGrid(theta));
			estimated.applyToGrid(filteredSinogramEst.getSubGrid(theta));
		}
		filteredSinogram.show("The Filtered Sinogram");
		estimated.show("Estimated");
		// Backproject and show
		
		backproj.backprojectPixelDriven(filteredSinogram).show("The Reconstruction Ram Lak");
		backproj.backprojectPixelDriven(filteredSinogramRamp).show("The Reconstruction Ramp");
		backproj.backprojectPixelDriven(filteredSinogramEst).show("The Reconstruction Esitmated");
		
	}

}

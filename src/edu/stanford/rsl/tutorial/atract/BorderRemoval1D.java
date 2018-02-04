package edu.stanford.rsl.tutorial.atract;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;

/**
 * This class implements a method to remove the borders caused by the Collimator. 
 * Use this for 1-D detectors.
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class BorderRemoval1D {

	
	/**
	 * Use this method in combination with the Laplace2D Filter.
	 * This method searches for the collimator borders in a laplacian filtered image.
	 * The current implementation searches in each detector for the first and last non-zero element.
	 * The coordinates with the maximum number of non-zero elements across all projections are then assumed to be the 
	 * borders and set to zero.
	 * @param sino Laplacian filtered Sinogram
	 */
	public void applyToGridLap2D(Grid2D sino) {
		double eps = 0.1;
		int maxThetaIndex = sino.getSize()[1];
		int maxSIndex = sino.getSize()[0];
		int counter[] = new int[maxSIndex];
		for(int i = 0; i < maxThetaIndex; i++) {
			for(int j = 0; j < maxSIndex/2; j++) {
				if(sino.getAtIndex(j,i)*-1.0 > eps) {
					counter[j]++;
					break;
				}
			}
			for(int j =  maxSIndex-1; j > maxSIndex/2; j--) {
				if(sino.getAtIndex(j,i)*-1.0 > eps ) {
					counter[j]++;
					break;
				}
			}
		}
		
		int tmp1  = 0;
		int max1 = 0;
		for(int i = 0; i < maxSIndex/2; i++) {
			if(counter[i] > max1) {
				max1 = counter[i];
				tmp1 = i;
			}
		}
		int tmp2 = 0;
		int max2 = 0;
		for(int j =  maxSIndex-1; j > maxSIndex/2; j--) {
			if(counter[j] > max2) {
				max2 = counter[j];
				tmp2 = j;
			}
		}
		for(int i = 0; i < maxThetaIndex; i++) {
			sino.setAtIndex( tmp1,i, 0);
			sino.setAtIndex(tmp1+1,i, 0);
			sino.setAtIndex(tmp2,i, 0);
			sino.setAtIndex(tmp2-1,i, 0);
		}
		
	}
	/**
	 * Use this method in combination with the Laplace1D Filter.
	 * This method searches for the collimator borders in a laplacian filtered image.
	 * The current implementation searches in each detector for the first and last non-zero element.
	 * The coordinates with the maximum number of non-zero elements across all projections are then assumed to be the 
	 * borders and set to zero.
	 * @param sino Laplacian filtered Sinogram
	 */
	public void applyToGridLap1D(Grid2D sino) {
		double eps = 0.1;
		int maxThetaIndex = sino.getSize()[1];
		int maxSIndex = sino.getSize()[0];
		int counter[] = new int[maxSIndex];
		for(int i = 0; i < maxThetaIndex; i++) {
			for(int j = 0; j < maxSIndex/2; j++) {
				if(sino.getAtIndex(j,i) > eps) {
					counter[j]++;
					break;
				}
			}
			for(int j =  maxSIndex-1; j > maxSIndex/2; j--) {
				if(sino.getAtIndex(j,i) > eps ) {
					counter[j]++;
					break;
				}
			}
		}
		
		int tmp1  = 0;
		int max1 = 0;
		for(int i = 0; i < maxSIndex/2; i++) {
			if(counter[i] > max1) {
				max1 = counter[i];
				tmp1 = i;
			}
		}
		int tmp2 = 0;
		int max2 = 0;
		for(int j =  maxSIndex-1; j > maxSIndex/2; j--) {
			if(counter[j] > max2) {
				max2 = counter[j];
				tmp2 = j;
			}
		}
		for(int i = 0; i < maxThetaIndex; i++) {
			sino.setAtIndex( tmp1,i, 0);
			sino.setAtIndex(tmp1+1,i, 0);
			sino.setAtIndex(tmp2,i, 0);
			sino.setAtIndex(tmp2-1,i, 0);
		}
		
	}
	/**
	 * Use this method in combination with the Laplace1D Filter
	 * This method searches for the collimator borders in a laplacian filtered image.
	 * The current implementation searches in each detector for the first and last non-zero element.
	 * The coordinates with the maximum number of non-zero elements across all projections are then assumed to be the 
	 * borders and set to zero.
	 * @param input Laplacian filtered Sinogram
	 */
	public void applyToGridLap1D(Grid3D input){
		int iter = input.getSize()[2];
		
		for(int i = 0; i < iter; i++) {
			applyToGridLap1D(input.getSubGrid(i));
		}
	}
	/**
	 * Use this method in combination with the Laplace2D Filter
	 * This method searches for the collimator borders in a laplacian filtered image.
	 * The current implementation searches in each detector for the first and last non-zero element.
	 * The coordinates with the maximum number of non-zero elements across all projections are then assumed to be the 
	 * borders and set to zero.
	 * @param input Laplacian filtered Sinogram
	 */
	public void applyToGridLap2D(Grid3D input){
		int iter = input.getSize()[2];
		
		for(int i = 0; i < iter; i++) {
			applyToGridLap2D(input.getSubGrid(i));
		}
	}
	

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
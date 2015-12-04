/*
 * Copyright (C) 2014 Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.fourierConsistency.wedgefilter;

/**
 * This class creates a binary representation of the double wedge region of the Fourier transform
 * of a parallel beam sinogram. The output Grid2D can be used as band-stop filter. The implementation
 * to build the filter follows the mathematics described in the paper:
 * "Novel properties of the Fourier decomposition of the sinogram", 
 * Edholm, Paul R., Robert M. Lewitt, and Bernt Lindholm, Physics and Engineering of Computerized 
 * Multidimensional Imaging and Processing. International Society for Optics and Photonics, 1986.
 *  
 * @author Marcel Pohlmann
 * 
 */

public class DoubleWedgeFilterParallel extends DoubleWedgeFilter {
	
	/**
	 * Constructor for the parallel beam double wedge filter.
	 * 
	 * @param size Size of the 2D-Fourier transform
	 * @param spacing Spacing of the 2D-Fourier transform
	 * @param origin Origin of the 2D-Fourier transform
	 * @param rp Maximum object extend
	 */
	public DoubleWedgeFilterParallel(int[] size, double[] spacing, double[] origin, double rp) {
		super(size, spacing, origin, rp);
		
		this.update();
	}
	
	protected void update() {
		double omega, n;
		
		// computes the parallel beam double wedge filter following the parametrization
		// shown in Eq. (15) in the paper
		for(int i = 0; i < size[1]; ++i) {
			for(int j = 0; j < size[0]; ++j) {
				omega = this.indexToPhysical(j, i)[0];
				n = this.indexToPhysical(j, i)[1];
				
				if (Math.abs(n/omega) > (this.rp)) {
					this.setAtIndex(j, i, 0.0f);
				}else{
					this.setAtIndex(j, i, 1.0f);
				}
			}
		}
	}

}

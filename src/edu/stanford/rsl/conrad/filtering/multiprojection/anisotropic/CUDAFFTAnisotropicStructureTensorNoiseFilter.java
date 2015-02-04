package edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic;

import edu.stanford.rsl.conrad.cuda.CUDAFFTVolumeHandle;
import edu.stanford.rsl.conrad.volume3d.ParallelVolumeOperator;


/**
 * MultiProjectionFilter which implements an anisotropic structure tensor noise filter.
 * FFT is computed using CUDA.
 * 
 * @author akmaier
 *
 */
public class CUDAFFTAnisotropicStructureTensorNoiseFilter extends
		AnisotropicStructureTensorNoiseFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5714621818252161125L;

	@Override
	public String getToolName() {
		return "CUDA FFT Anisotropic Structure Tensor Noise Filter";
	}
	
	@Override
	protected AnisotropicFilterFunction getAnisotropicFilterFunction(){
		return new AnisotropicFilterFunction(new CUDAFFTVolumeHandle(new ParallelVolumeOperator()), new ParallelVolumeOperator());
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
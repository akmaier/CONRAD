package edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic;

import jcuda.runtime.JCuda;
import edu.stanford.rsl.conrad.cuda.CUDAFFTVolumeHandle;
import edu.stanford.rsl.conrad.cuda.CUDAVolume3D;
import edu.stanford.rsl.conrad.cuda.CUDAVolumeOperator;
import edu.stanford.rsl.conrad.volume3d.Volume3D;

/**
 * MultiProjectionFilter which implements an anisotropic structure tensor noise filter on CUDA.
 * 
 * @author akmaier
 *
 */
public class CUDAAnisotropicStructureTensorNoiseFilter extends
		AnisotropicStructureTensorNoiseFilter {

	
	CUDAVolumeOperator cuop = null;
	/**
	 * 
	 */
	private static final long serialVersionUID = 5714621818252161125L;

	@Override
	public String getToolName() {
		return "CUDA Anisotropic Structure Tensor Noise Filter";
	}
	
	@Override
	protected AnisotropicFilterFunction getAnisotropicFilterFunction(){
		cuop = new CUDAVolumeOperator();
		cuop.initCUDA();
		return new AnisotropicFilterFunction(new CUDAFFTVolumeHandle(cuop), cuop);
	}
	
	@Override
	protected void fetchImageData(Volume3D [] filtered){
		((CUDAVolume3D) filtered[0]).fetch();
		if (filtered[1] !=null) ((CUDAVolume3D) filtered[1]).fetch();
	}
	
	@Override 
	protected void cleanup(){
		cuop.cleanup();
		JCuda.cudaThreadExit();
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
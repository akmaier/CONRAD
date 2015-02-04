package edu.stanford.rsl.conrad.volume3d;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_3D;
import edu.emory.mathcs.utils.ConcurrencyUtils;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class JTransformsFFTVolumeHandle extends FFTVolumeHandle {

	public JTransformsFFTVolumeHandle(VolumeOperator operator){
		super(operator);
	}
	
	@Override
	public void forwardTransform(Volume3D vol)
	{
		if (debug)
			System.out.println("vol_fft\n");

		operator.makeComplex(vol);  
		CONRAD.gc();

		FloatFFT_3D fft = new FloatFFT_3D(vol.size[0], vol.size[1], vol.size[2]);
		
		fft.complexForward(vol.data);


	}

	@Override
	public void inverseTransform(Volume3D vol)
	{
		if (debug)
			System.out.println("JTransforms vol_ifft\n");

		operator.makeComplex(vol); 
		CONRAD.gc();

		FloatFFT_3D fft = new FloatFFT_3D(vol.size[0], vol.size[1], vol.size[2]);
		
		fft.complexInverse(vol.data, true);
		
	}

	@Override
	public void cleanUp() {
		// Nothing to do here
		
	}

	@Override
	public void setThreadNumber(int number) {
		ConcurrencyUtils.setNumberOfThreads(number);
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
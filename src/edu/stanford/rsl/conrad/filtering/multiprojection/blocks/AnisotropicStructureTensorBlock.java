package edu.stanford.rsl.conrad.filtering.multiprojection.blocks;

import ij.ImagePlus;
import edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.volume3d.JTransformsFFTVolumeHandle;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import edu.stanford.rsl.conrad.volume3d.VolumeOperator;


/**
 * Class implements block-wise processing of the anisotropic structure tensor.
 * This implementation scales better with an increasing number of CPUs.
 * If computation on the graphics card is desired, use the MultiProjectionFilter implementation based on CUDAVolume3D.
 * 
 * @see edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction
 * @see edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicStructureTensorNoiseFilter
 * @see edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.CUDAAnisotropicStructureTensorNoiseFilter
 * @author akmaier
 *
 */
public class AnisotropicStructureTensorBlock extends ImageProcessingBlock {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1265388205607681586L;
	protected float smoothness = 2.0f;
	protected float lowerTensorLevel = 0.77f;
	protected float upperTensorLevel = 1.0f;
	protected float highPassLowerLevel = 0.0f;
	protected float highPassUpperLevel = 1.0f;
	protected float lpUpper = 1.5f;
	protected boolean showAbsoluteTensor = false;
	protected double dimx;
	protected double dimy;
	
	@Override
	public ImageProcessingBlock clone() {
		AnisotropicStructureTensorBlock clone =  new AnisotropicStructureTensorBlock();
		clone.smoothness = smoothness;
		clone.lowerTensorLevel = this.lowerTensorLevel;
		clone.upperTensorLevel = this.upperTensorLevel;
		clone.highPassLowerLevel = this.highPassLowerLevel;
		clone.highPassUpperLevel = this.highPassUpperLevel;
		clone.lpUpper = this.lpUpper;
		clone.dimx = dimx;
		clone.dimy = dimy;
		return clone;
	}

	@Override
	protected void processImageBlock() {
		// Convert given Grid3D to ImagePlus
		ImagePlus input = ImageUtil.wrapGrid3D(inputBlock, "Anisotropic Structure Tensor Block");
		
		AnisotropicFilterFunction filter = new AnisotropicFilterFunction(new JTransformsFFTVolumeHandle(new VolumeOperator()), new VolumeOperator());
		filter.setThreadNumber(1);
		boolean uneven = (input.getStackSize()*2) % 2 == 1;
		int margin = 15/2;
		if (margin%2==1) margin ++;
		Volume3D vol = filter.getVolumeOperator().createVolume(input, margin, 3,uneven);
		Volume3D [] filtered = filter.computeAnisotropicFilteredVolume(vol, lowerTensorLevel, upperTensorLevel, highPassLowerLevel, highPassUpperLevel, (float) smoothness, 1, 2.0f, 1.5f, 1.0f, lpUpper);
		
		ImagePlus output = filtered[0].getImagePlus("Anisotropic Filtered " + input.getTitle(), margin, 3,uneven);
		// Convert result back to Grid3D
		outputBlock = ImageUtil.wrapImagePlus(output);
		
		filter = null;
		filtered[0].destroy();
		if (filtered[1] != null) filtered[1].destroy();
		filtered = null;
		vol.destroy();
		vol = null;
	}

	public void configure() throws Exception {
		dimx = Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX();
		dimy = Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionY();
		smoothness = (float) UserUtil.queryDouble("Smoothness: ", smoothness);
		lowerTensorLevel = (float) UserUtil.queryDouble("Lower limit in Tensor: ", lowerTensorLevel);
		upperTensorLevel = (float) UserUtil.queryDouble("Upper limit in Tensor: ", upperTensorLevel);
		highPassUpperLevel = (float) UserUtil.queryDouble("Upper limit in high pass sigmoid: ", highPassUpperLevel);
		highPassLowerLevel = (float) UserUtil.queryDouble("Lower limit in high pass sigmoid: ", highPassLowerLevel);
		lpUpper = (float) UserUtil.queryDouble("Strengh of low pass filter: ", lpUpper);
		configured = true;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
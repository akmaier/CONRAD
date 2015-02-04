package edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.MultiProjectionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.volume3d.JTransformsFFTVolumeHandle;
import edu.stanford.rsl.conrad.volume3d.ParallelVolumeOperator;
import edu.stanford.rsl.conrad.volume3d.Volume3D;

/**
 * MultiProjectionFilter which implements an anisotropic structure tensor noise filter.
 * 
 * @author akmaier
 *
 */
public class AnisotropicStructureTensorNoiseFilter extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7877365673267053836L;
	protected float smoothness = 2.0f;
	public float getSmoothness() {
		return smoothness;
	}

	public void setSmoothness(float smoothness) {
		this.smoothness = smoothness;
	}

	public float getLowerTensorLevel() {
		return lowerTensorLevel;
	}

	public void setLowerTensorLevel(float lowerTensorLevel) {
		this.lowerTensorLevel = lowerTensorLevel;
	}

	public float getUpperTensorLevel() {
		return upperTensorLevel;
	}

	public void setUpperTensorLevel(float upperTensorLevel) {
		this.upperTensorLevel = upperTensorLevel;
	}

	public float getHighPassLowerLevel() {
		return highPassLowerLevel;
	}

	public void setHighPassLowerLevel(float highPassLowerLevel) {
		this.highPassLowerLevel = highPassLowerLevel;
	}

	public float getHighPassUpperLevel() {
		return highPassUpperLevel;
	}

	public void setHighPassUpperLevel(float highPassUpperLevel) {
		this.highPassUpperLevel = highPassUpperLevel;
	}

	public float getLpUpper() {
		return lpUpper;
	}

	public void setLpUpper(float lpUpper) {
		this.lpUpper = lpUpper;
	}

	public boolean isShowAbsoluteTensor() {
		return showAbsoluteTensor;
	}

	public void setShowAbsoluteTensor(boolean showAbsoluteTensor) {
		this.showAbsoluteTensor = showAbsoluteTensor;
	}

	public boolean isOverlap() {
		return overlap;
	}

	public void setOverlap(boolean overlap) {
		this.overlap = overlap;
	}

	protected float lowerTensorLevel = 0.77f;
	protected float upperTensorLevel = 1.0f;
	protected float highPassLowerLevel = 0.0f;
	protected float highPassUpperLevel = 1.0f;
	protected float lpUpper = 1.5f;
	protected boolean showAbsoluteTensor = false;
	protected double dimx;
	protected double dimy;
	protected ImageGridBuffer tensorBuffer;
	protected boolean overlap = false;

	@Override
	public void prepareForSerialization(){
		tensorBuffer = null;
	}

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		if (tensorBuffer == null) tensorBuffer = new ImageGridBuffer();
		int upperEnd = upperEnd(projectionNumber);
		int lowerEnd = lowerEnd(projectionNumber);
		int newBlockCenter1 = upperEnd - context;
		int newBlockCenter2 = upperEnd - (context/2);
		int blockCenter  = (projectionNumber / context) * context + (context/2);
		if (!overlap){
			blockCenter  = (projectionNumber / (context * 2 + 1)) * (context * 2 + 1) + (context);
		}
		boolean process = false;
		if (!isLastBlock(projectionNumber)) { // All other blocks have regular length
			process = (projectionNumber - blockCenter) == 0;
		} else { // Last block.
			if (debug > 2) System.out.println("AnisotropicStructureTensorNoiseFilter: " + projectionNumber + " " + newBlockCenter1);
			blockCenter = newBlockCenter1;
			if (!overlap){
				newBlockCenter2 = newBlockCenter1;
			}
			if (projectionNumber == newBlockCenter2){
				blockCenter = newBlockCenter2;
				lowerEnd = upperEnd - (context*2 +1);
				if (lowerEnd < 0){
					lowerEnd = 0;
				}
			}
			process = (projectionNumber == newBlockCenter1) || (projectionNumber == newBlockCenter2);
		}
		if (process) {
			if (debug>0) System.out.println("Processing block centered at: "+ projectionNumber + " "  + System.currentTimeMillis());
			ImagePlus current = new ImagePlus();
			Grid2D grid = inputQueue.get(projectionNumber);
			FloatProcessor model = new FloatProcessor(grid.getWidth(), grid.getHeight(), grid.getBuffer());
			int width = model.getWidth();
			int height = model.getHeight();
			ImageStack stack = new ImageStack(width, height, upperEnd-lowerEnd);
			for (int i = lowerEnd; i < upperEnd; i++){
				stack.setPixels(inputQueue.get(i).getPixels(), i - lowerEnd + 1);
			}
			current.setStack("To Filter", stack);	
			try{
				ImagePlus [] result = filterAnisotropic(current);
				// Write into the output Queue
				int lowerBound = Math.max(lowerEnd, blockCenter-(context/2));
				if (!overlap) {
					lowerBound = lowerEnd;
				}
				//result[0].show();
				
				for (int i = lowerBound; i < upperEnd; i++){
					FloatProcessor processor = (FloatProcessor) result[0].getStack().getProcessor(i-lowerEnd +1);
					outputQueue.add(new Grid2D((float[])processor.getPixels(), processor.getWidth(), processor.getHeight()), i);
					if (showAbsoluteTensor){
						FloatProcessor processor2 = (FloatProcessor) result[0].getStack().getProcessor(i-lowerEnd +1);
						tensorBuffer.add(new Grid2D((float[])processor2.getPixels(), processor2.getWidth(), processor2.getHeight()), i);
					}
				}
				// Collect Garbage
				CONRAD.gc();
			} catch (Exception e){
				e.printStackTrace();
			}
			// outputQueue should filter double projections.
			if (isLastBlock(projectionNumber)&& (projectionNumber == newBlockCenter2)){
				for (int i = 0; i < outputQueue.size(); i++){
					Grid2D image = outputQueue.get(i);
					if (debug >2) System.out.println("Ani: " + i + " " + image);
					sink.process(image, i);
				}
				if (showAbsoluteTensor) {
					tensorBuffer.toImagePlus("Magnitude of Structure Tensor").show();
				}
			}
			if (debug > 0) System.out.println("End of block: " + System.currentTimeMillis());
		}

	}
	
	protected ImagePlus [] filterAnisotropic(ImagePlus current){
		boolean uneven = (current.getStackSize()*2) % 2 == 1;
		int margin = context/2;
		if (margin%2==1) margin ++;
		AnisotropicFilterFunction filter = getAnisotropicFilterFunction();
		Volume3D vol = filter.getVolumeOperator().createVolume(current, margin, 3,uneven);
		Volume3D [] filtered = filter.computeAnisotropicFilteredVolume(vol, lowerTensorLevel, upperTensorLevel, highPassLowerLevel, highPassUpperLevel, (float) smoothness, 1, 2.0f, 1.5f, 1.0f, lpUpper);
		ImagePlus [] images = new ImagePlus[2];
		fetchImageData(filtered);
		if (debug > 0) System.out.println("Filtering Step done");
		images[0] = filtered[0].getImagePlus("Anisotropic Filtered " + current.getTitle(), margin, 3,uneven);
		if (showAbsoluteTensor){
			images[1] = filtered[1].getImagePlus("Tensor Segment " + current.getTitle(), margin, 3,uneven);
		}
		filter = null;
		filtered[0].destroy();
		if (filtered[1] != null) filtered[1].destroy();
		filtered = null;
		vol.destroy();
		vol = null;
		cleanup();
		return images;
	}
	
	protected void fetchImageData(Volume3D[] filtered) {
		// nothing to do
	}

	protected void cleanup(){
		CONRAD.gc();
	}
	
	protected AnisotropicFilterFunction getAnisotropicFilterFunction(){
		return new AnisotropicFilterFunction(new JTransformsFFTVolumeHandle(new ParallelVolumeOperator()), new ParallelVolumeOperator());
	}

	@Override
	public ImageFilteringTool clone() {
		AnisotropicStructureTensorNoiseFilter clone = null;
		try {
			clone = this.getClass().newInstance();
			clone.smoothness = smoothness;
			clone.context = context;
			clone.lowerTensorLevel = lowerTensorLevel;
			clone.upperTensorLevel = upperTensorLevel;
			clone.highPassLowerLevel = highPassLowerLevel;
			clone.highPassUpperLevel = highPassUpperLevel;
			clone.lpUpper = lpUpper;
			clone.showAbsoluteTensor = showAbsoluteTensor;
			clone.dimx = dimx;
			clone.dimy = dimy;
			clone.overlap = overlap;
			clone.configured = configured;
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		}
		return clone;
	}

	@Override
	public String getToolName() {
		return "Anisotropic Structure Tensor Noise Filter";
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	public void configure() throws Exception {
		dimx = Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX();
		dimy = Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionY();
		context = UserUtil.queryInt("Segment Size: ", context);
		smoothness = (float) UserUtil.queryDouble("Smoothness: ", smoothness);
		lowerTensorLevel = (float) UserUtil.queryDouble("Lower limit in Tensor: ", lowerTensorLevel);
		upperTensorLevel = (float) UserUtil.queryDouble("Upper limit in Tensor: ", upperTensorLevel);
		highPassUpperLevel = (float) UserUtil.queryDouble("Upper limit in high pass sigmoid: ", highPassUpperLevel);
		highPassLowerLevel = (float) UserUtil.queryDouble("Lower limit in high pass sigmoid: ", highPassLowerLevel);
		lpUpper = (float) UserUtil.queryDouble("Strengh of low pass filter: ", lpUpper);
		showAbsoluteTensor = UserUtil.queryBoolean("Display Magnitude Image of Tensor?");
		overlap = UserUtil.queryBoolean("Compute with overlap?");
		configured = true;
	}

	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	public double getDimx() {
		return dimx;
	}

	public void setDimx(double dimx) {
		this.dimx = dimx;
	}

	public double getDimy() {
		return dimy;
	}

	public void setDimy(double dimy) {
		this.dimy = dimy;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

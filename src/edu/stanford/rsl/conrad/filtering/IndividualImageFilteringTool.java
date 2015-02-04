package edu.stanford.rsl.conrad.filtering;

import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.parallel.NamedParallelizableRunnable;

public abstract class IndividualImageFilteringTool extends ImageFilteringTool implements NamedParallelizableRunnable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2504582813925296457L;
	protected Grid2D imageProcessor;
	protected CountDownLatch latch;
	protected int imageIndex;
	protected Grid2D theFiltered = null;
	
	/**
	 * All ImageFilteringTools need to be Cloneable in order to enable multiple processor usage.
	 */
	public abstract IndividualImageFilteringTool clone();
	
	/**
	 * Returns the index of the current image in the ImageStack. 
	 * 
	 * @see #setImageIndex(int)
	 * @return the current image index
	 */
	public int getImageIndex() {
		return imageIndex;
	}
	
	/**
	 * Is called in every thread to apply the tool to an individual ImageProcessor
	 * @param imageProcessor the ImageProcessor
	 * @return the filtered instance of the ImageProcessor
	 * @throws Exception
	 */
	public abstract Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception;

	/**
	 * Sets the number of the current image in the stack. This information my be required for some filters
	 * which apply a slightly different filter to each image. This depends on the actual implementation of
	 * the filter. The AECFilter for example requires this information.
	 * 
	 * @param imageIndex the index of the current image.
	 */
	public void setImageIndex(int imageIndex) {
		this.imageIndex = imageIndex;
	}

	/**
	 * Sets the actual ImageProcessor to be filtered. A duplicate will be created for each thread.
	 * 
	 * @param imageProcessor the processor to be filtered.
	 */
	public void setImageProcessor(Grid2D imageProcessor) {
		// FIXME ?
		if (imageProcessor != null){
			this.imageProcessor = imageProcessor;//.duplicate();
		} else {
			this.imageProcessor = null;
		}
	}

	@Override
	/**
	 * Sets the CountDownLatch which is countDown() once the processing is finished.
	 */
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}
	
	/**
	 * This method performs the correction on one frame, i.e., one projection image. 
	 */
	@Override
	public void run() {
		try {
			theFiltered = applyToolToImage(imageProcessor);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		latch.countDown();
	}
	
	
	/**
	 * returns the filtered ImageProcessor if the filtering was successful.
	 * This getter violates the getter setter paradigm on purpose.
	 * @return the filtered ImageProcessor
	 */
	public Grid2D getFilteredImage() {
		return theFiltered;
	}
	
	@Override
	public void prepareForSerialization(){
		imageProcessor = null;
		theFiltered = null;
	}
	
	public String getProcessName(){
		return getToolName();
	}
	

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

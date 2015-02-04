package edu.stanford.rsl.conrad.filtering.multiprojection.blocks;

import java.io.Serializable;
import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.parallel.ParallelizableRunnable;

public abstract class ImageProcessingBlock implements ParallelizableRunnable, Cloneable, GUIConfigurable, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5847536522784571557L;
	protected CountDownLatch latch;
	protected Grid3D inputBlock;
	protected Grid3D outputBlock;
	protected boolean configured = false;
	
	public void run(){
		processImageBlock();
		latch.countDown();
	}
	
	
	/**
	 * Method to process the current image block. Is called by run() which is called by a BlockWiseMultiProjectionFilter to run the blocks in parallel.
	 */
	protected abstract void processImageBlock();
	
	
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}


	public Grid3D getInputBlock() {
		return inputBlock;
	}


	public void setInputBlock(Grid3D inputBlock) {
		this.inputBlock = inputBlock;
	}


	public Grid3D getOutputBlock() {
		return outputBlock;
	}


	public void setOutputBlock(Grid3D outputBlock) {
		this.outputBlock = outputBlock;
	}
	
	@Override
	public abstract ImageProcessingBlock clone();

	public boolean isConfigured() {
		return configured;
	}
	
	public void prepareForSerialization(){
		inputBlock = null;
		outputBlock = null;
		latch = null;
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
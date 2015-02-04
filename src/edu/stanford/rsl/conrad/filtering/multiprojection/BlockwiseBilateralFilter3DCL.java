package edu.stanford.rsl.conrad.filtering.multiprojection;

import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.blocks.BilateralFilter3DCLBlock;
import edu.stanford.rsl.conrad.filtering.opencl.BilateralFiltering3DTool;

/**
 * Class for computation of Bilateral Filter which is based on OpenCL. Computation is carried out in several threads.
 * @author Benedikt Lorch
 *
 */
public class BlockwiseBilateralFilter3DCL extends BlockWiseMultiProjectionFilter {

	private static final long serialVersionUID = 2430924392909104202L;
	protected final double PERCENTAGE_OF_MEM_TO_BE_USED = 0.95f;

	@Override
	public ImageFilteringTool clone() {
		BlockwiseBilateralFilter3DCL clone = new BlockwiseBilateralFilter3DCL();
		clone.modelBlock = modelBlock;
		clone.numBlocks = numBlocks;
		clone.blockOverlap = blockOverlap;
		clone.configured = configured;
		return clone;
	}
	

	@Override
	public String getToolName() {
		return "OpenCL Blockwise Bilateral Filter 3-D";
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Tomasi98-BFF,\n" +
		"  author = {Tomasi, C. and Manduchi, R.},\n" +
		"  title = {Bilateral Filtering for Gray and Color Images},\n" +
		"  booktitle = {ICCV '98: Proceedings of the Sixth International Conference on Computer Vision},\n" +
		"  year = {1998},\n" +
		"  isbn = {81-7319-221-9},\n" +
		"  pages = {839-846},\n" +
		"  publisher = {IEEE Computer Society},\n" +
		"  address = {Washington, DC, USA},\n" +
		"}\n";
		return bibtex;
	}
	
	
	/**
	 * Configures the model block. The configuring values for the superior BlockWiseMultiProjectionFilter will be computed according to the kernel width and the device's memory space
	 * @throws Exception
	 */
	public void configure() throws Exception {
		
		if (modelBlock == null) {
			modelBlock = new BilateralFilter3DCLBlock();
		}
	
		// Get the block's filtering tool ...
		BilateralFiltering3DTool bilateralFilteringTool = (BilateralFiltering3DTool) ((BilateralFilter3DCLBlock)modelBlock).getFilteringTool();
		bilateralFilteringTool.setAskForGuidance(false);

		modelBlock.configure();
		
		configured = true;
	}
	
	
	public void configure(double sigmaGeom[], double sigmaPhotom) throws Exception {
		
		if (null == modelBlock) {
			modelBlock = new BilateralFilter3DCLBlock();
		}
		
		BilateralFiltering3DTool bilateralFilteringTool = (BilateralFiltering3DTool) ((BilateralFilter3DCLBlock)modelBlock).getFilteringTool();
		bilateralFilteringTool.setAskForGuidance(false);
		bilateralFilteringTool.setSigmaGeom(sigmaGeom);
		bilateralFilteringTool.setSigmaPhoto(sigmaPhotom);
		bilateralFilteringTool.setKernelWidth();
		bilateralFilteringTool.setConfigured(true);
		
		configured = true;
	}
	
	/**
	 * Initialize blocks like the super class does, but calculate block overlap with data from inputQueue first
	 */
	@Override
	protected synchronized void initBlocks (){
		if (!initBlocks) {
			calculateBlockOverlapAndNumber();
			super.initBlocks();
		}
	}
	
	
	/**
	 * Calculate the block overlap and the number of blocks according to kernelWidth and the device's memory size 
	 */
	protected void calculateBlockOverlapAndNumber() {
		// Instead of calculating block overlap and number of blocks, these values can be queried manually using
		// super.configure();
		
		int iWidth = inputQueue.get(0).getWidth();
		int iHeight = inputQueue.get(0).getHeight();
		int iDepth = inputQueue.size();
		
		// Get the block's filtering tool
		BilateralFiltering3DTool bilateralFilteringTool = (BilateralFiltering3DTool) ((BilateralFilter3DCLBlock)modelBlock).getFilteringTool();
		
		// Calculate number of blocks
		CLDevice device = bilateralFilteringTool.getDevice();
		
		// Get the global memory size. This gives the number of bytes.
		// E.g. device with 4GB may return 4 294 770 688
		long globalMemSize = device.getGlobalMemSize();
		
		// Don't use 100 %
		long bytesToUse = (long) Math.floor(PERCENTAGE_OF_MEM_TO_BE_USED * globalMemSize);
		
		// 1 float takes 4 bytes (IEEE 754)		
		long imageSize = 4 * iWidth * iHeight * iDepth;
		// We need to copy two buffers onto the processing device: input and output
		long totalSize = 2 * imageSize;
		
		int blocks = (int) Math.ceil(totalSize / ((double) bytesToUse));
		if (blocks > 1) {
			// Use am even number of blocks
			if (blocks % 2 != 0) {
				blocks++;
			}
			
			// block overlap can remain 0 if we only use one block
			int kernelWidth = bilateralFilteringTool.getKernelWidth();
			int[] blockOverlap = { kernelWidth/2, kernelWidth/2, kernelWidth/2 };
			super.blockOverlap = blockOverlap;
		}
		
		
		super.numBlocks = blocks;
		
		if (debug > 0) {
			System.out.println("Using " + numBlocks + " blocks with block overlap [" + super.blockOverlap[0] + ", " + super.blockOverlap[1] + ", " + super.blockOverlap[2] + "]");
		}
	}
	

	@Override
	public String getMedlineCitation() {
		return "Tomasi C, Maduchi R, Bilateral Filtering for Gray and Color Images. In: ICCV '98: Proceedings of the Sixth International Conference on Computer Vision, pp. 839-846, IEEE Computer Society, Washington, DC, United States 1998.";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
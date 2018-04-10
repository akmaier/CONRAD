/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.opencl;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLMemory;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;

/**
 * Tool for OpenCL computation of the Mean Filter in a Grid 3D using the 6-connected neighborhood of each voxel.
 * This tool uses the CONRAD internal Grid 3-D data structure.
 * 
 * If desired, not all voxels in the input Grid3D are processed. Then, a Grid3D of the same size as the processed image
 * has to be handed to the tool using the Setter for computeValueGrid. This computeValueGrid contains zeros for all
 * voxels that should NOT be processed. If no computeValueGrid is handed to the tool, all voxels are processed. 
 *  
 * @author Jennifer Maier
 *
 */

public class MeanFiltering3DTool extends OpenCLFilteringTool3D {
	
	private static final long serialVersionUID = -1469783337390366468L;
	protected CLBuffer<FloatBuffer> computeValue;
	private Grid3D computeValueGrid;

	public MeanFiltering3DTool() {
		this.kernelName = kernelname.MEAN_FILTER_3D;
	}

	@Override
	public void configure() throws Exception {		
		
		this.kernelName = kernelname.MEAN_FILTER_3D;
				
		// if the opneCL structures are initialized and the computeValueGrid is set, we can already create the openCL
		// computeValue Buffer
		if (init && computeValueGrid != null) {
			computeValue = clContext.createFloatBuffer(width * height * depth, CLMemory.Mem.READ_ONLY);
			gridToBuffer(computeValue.getBuffer(), computeValueGrid);
			computeValue.getBuffer().rewind();
		}

		configured = true;
		
	}
	
	/**
	 * Called by process() before the processing begins. Put your write buffers to the queue here.
	 * @param input Grid 3-D to be processed
	 * @param queue CommandQueue for the specific device
	 */
	@Override
	protected void prepareProcessing(Grid3D input, CLCommandQueue queue) {
				
		// Copy image data into linear floatBuffer
		gridToBuffer(image.getBuffer(), input);
		image.getBuffer().rewind();
		queue.putWriteBuffer(image, true);		
		
		// initialize openCL computeValue Buffer, if this hasn't been done before and put it in queue
		if (computeValue == null) {
			// if no computeValueGrid was set, all voxels should be processed
			// -> Grid3D of same size as input containing only non-zero values
			if (computeValueGrid == null) {
				computeValueGrid = new Grid3D(width, height, depth);
				NumericPointwiseOperators.fill(computeValueGrid, 1.0f);
			}
			computeValue = clContext.createFloatBuffer(width * height * depth, CLMemory.Mem.READ_ONLY);
			gridToBuffer(computeValue.getBuffer(), computeValueGrid);
			computeValue.getBuffer().rewind();
		}
		queue.putWriteBuffer(computeValue, true);
		
	}
	
	// Getter and Setter
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
	public void setComputeValueGrid (Grid3D computeValueGrid) {
		this.computeValueGrid = computeValueGrid;
	}

	public Grid3D getComputeValueGrid () {
		return this.computeValueGrid;
	}
	
	@Override
	protected void configureKernel() {
		filterKernel = program.createCLKernel("meanFilter6Surrounding3D");
		
		filterKernel
		.putArg(image)
		.putArg(result)
		.putArg(computeValue)
		.putArg(width)
		.putArg(height)
		.putArg(depth);
		
	}
	
	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getToolName() {
		return "OpenCL Mean Filter 3D (6-connected)";
	}

	@Override
	public ImageFilteringTool clone() {
		MeanFiltering3DTool clone = new MeanFiltering3DTool();
		clone.setComputeValueGrid((Grid3D) this.computeValueGrid.clone());
		clone.setConfigured(this.configured);

		return clone;
	}

}

/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
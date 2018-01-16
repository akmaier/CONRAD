/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.opencl;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLMemory;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Tool for OpenCL Filtering of a Grid 2D with an arbitrary 3x3 Filter Kernel.
 * The values in the kernel should sum up to 1.
 * This tool uses the CONRAD internal Grid 2-D data structure.
 * 
 * @author Jennifer Maier
 *
 */
public class Kernel3x3Filtering2DTool extends OpenCLFilteringTool2D {

	private static final long serialVersionUID = -3381321981652625848L;
	protected double[] kernel = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
	private static final int KERNEL_WIDTH = 3;
	protected CLBuffer<FloatBuffer> kernelBuffer;
	
	public Kernel3x3Filtering2DTool() {
		this.kernelName = kernelname.KERNEL3x3_FILTER_2D;
	}
	
	@Override
	public void configure() throws Exception {		

		kernel = UserUtil.queryArray("Enter 3x3 kernel as one line", kernel);
		if (kernel.length != KERNEL_WIDTH*KERNEL_WIDTH) {
			throw new IllegalArgumentException("Kernel's length needs to be 9.");
		}

		configured = true;

	}
	
	/**
	 * Called by process() before the processing begins. Put your write buffers to the queue here.
	 * @param input Grid 3-D to be processed
	 * @param queue CommandQueue for the specific device
	 */
	@Override
	protected void prepareProcessing(Grid2D input, CLCommandQueue queue) {
		
		// Copy image data into linear floatBuffer
		gridToBuffer(image.getBuffer(), input);
		image.getBuffer().rewind();
		queue.putWriteBuffer(image, true);
		
		// prepare kernel and buffer
		float[] kernelFloat = {(float) kernel[0], (float) kernel[1], (float) kernel[2], (float) kernel[3],
				(float) kernel[4], (float) kernel[5], (float) kernel[6], (float) kernel[7], (float) kernel[8]};
		Grid2D kernelGrid = new Grid2D(kernelFloat, KERNEL_WIDTH, KERNEL_WIDTH);
		kernelBuffer = clContext.createFloatBuffer(KERNEL_WIDTH * KERNEL_WIDTH, CLMemory.Mem.READ_ONLY);
		gridToBuffer(kernelBuffer.getBuffer(), kernelGrid);
		kernelBuffer.getBuffer().rewind();
		queue.putWriteBuffer(kernelBuffer, true);
	}
	
	// Getter and Setter

	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
	public double[] getKernel() {
		return kernel;
	}

	public void setKernel(double[] kernel) {
		// check whether it is a 3x3 Kernel
		if (kernel.length != KERNEL_WIDTH*KERNEL_WIDTH) {
			throw new IllegalArgumentException("Kernel's length needs to be 9.");
		}
		this.kernel = kernel;
	}
	
	public void setKernel(double[][] kernel) {
		// check whether it is a 3x3 Kernel
		if (kernel.length * kernel[0].length != KERNEL_WIDTH*KERNEL_WIDTH) {
			throw new IllegalArgumentException("Kernel's length needs to be 9.");
		}
		
		// restructure double[][] to double[]
		double[] kernelArray = new double[KERNEL_WIDTH * KERNEL_WIDTH];
		for (int i = 0; i < kernel.length; i++) {
			for (int j = 0; j < kernel[0].length; j++) {
				kernelArray[i * kernel.length + j] = kernel[i][j];
			}
		}
		
		this.kernel = kernelArray;
	}
	
	@Override
	protected void configureKernel() {
		filterKernel = program.createCLKernel("kernel3x3Filter2D");

		filterKernel
		.putArg(image)
		.putArg(result)
		.putArg(width)
		.putArg(height)
		.putArg(KERNEL_WIDTH)
		.putArg(kernelBuffer);	

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
		return "OpenCL Filter 3x3 Kernel  2D";
	}

	@Override
	public ImageFilteringTool clone() {
		Kernel3x3Filtering2DTool clone = new Kernel3x3Filtering2DTool();
		clone.setKernel(this.getKernel().clone());
		clone.setConfigured(this.configured);
		return clone;
	}
	
}

/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
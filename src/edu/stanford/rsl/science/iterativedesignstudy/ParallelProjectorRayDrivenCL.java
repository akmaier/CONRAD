package edu.stanford.rsl.science.iterativedesignstudy;

import java.io.IOException;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class ParallelProjectorRayDrivenCL extends Projector{
	private double maxTheta;
	private double deltaTheta;
	private double maxS;
	private double deltaS;				// [mm]
	private int maxThetaIndex;
	private int maxSIndex;

	/**
	 * Sampling of projections is defined in the constructor.
	 * 
	 * @param maxTheta the angular range in radians
	 * @param deltaTheta the angular step size in radians
	 * @param maxS the detector size in [mm]
	 * @param deltaS the detector element size in [mm]
	 */
	public ParallelProjectorRayDrivenCL(double maxTheta, double deltaTheta, double maxS,double deltaS) {
		this.maxS = maxS;
		this.maxTheta = maxTheta;
		this.deltaS = deltaS;
		this.deltaTheta = deltaTheta;
		this.maxSIndex = (int) (maxS / deltaS + 1);
		this.maxThetaIndex = (int) (maxTheta / deltaTheta + 1);
	}
	@Override
	public NumericGrid project(NumericGrid grid, NumericGrid sino) {
		boolean debug = false;
		// create context
		CLContext context = OpenCLUtil.createContext();
		if (debug)
			System.out.println("Context: " + context);
		//show OpenCL devices in System
		CLDevice[] devices = context.getDevices();
		if (debug){
			for (CLDevice dev: devices)
				System.out.println(dev);
		}

		// select device
		CLDevice device = context.getMaxFlopsDevice();
		if (debug)
			System.out.println("Device: " + device);

		int imageSize = grid.getSize()[0] * grid.getSize()[1];
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxSIndex); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeBeta = OpenCLUtil.roundUp(localWorkSize, maxThetaIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ParallelProjectorRayDriven.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(imageSize, Mem.READ_ONLY);
		//		for (int i = 0; i < grid.getSize()[0]; ++i) {
		//			imageBuffer.getBuffer().put(grid.getSubGrid(i).getBuffer());
		//		}

		for (int i=0;i<grid.getSize()[1];++i){
			for (int j=0;j<grid.getSize()[0];++j)
				imageBuffer.getBuffer().put(((Grid2D)grid).getAtIndex(j, i));
		}
		imageBuffer.getBuffer().rewind();
		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(
				imageBuffer.getBuffer(), grid.getSize()[0], grid.getSize()[1],
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxSIndex * maxThetaIndex, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven2DCL");
		kernel.putArg(imageGrid).putArg(sinogram)
		.putArg((float)maxS).putArg((float)deltaS)
		.putArg((float)maxTheta).putArg((float)deltaTheta)
		.putArg(maxSIndex).putArg(maxThetaIndex); // TODO: Spacing :)

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteImage(imageGrid, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeBeta, globalWorkSizeT,
				localWorkSize, localWorkSize).putBarrier()
				.putReadBuffer(sinogram, true)
				.finish();

		// write sinogram back to grid2D
		//Grid2D sino = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
		sino.setSpacing(deltaS, deltaTheta);
		sinogram.getBuffer().rewind();
		for (int i = 0; i < ((Grid2D) sino).getBuffer().length; ++i) {
			((Grid2D) sino).getBuffer()[i] = sinogram.getBuffer().get();
		}

		// clean up
		queue.release();
		imageGrid.release();
		sinogram.release();
		kernel.release();
		program.release();
		context.release();

		return sino;
	}


	@Override
	public NumericGrid project(NumericGrid grid, NumericGrid sino, int index) {
		boolean debug = false;
		// create context
		CLContext context = OpenCLUtil.createContext();
		if (debug)
			System.out.println("Context: " + context);
		//show OpenCL devices in System
		CLDevice[] devices = context.getDevices();
		if (debug){
			for (CLDevice dev: devices)
				System.out.println(dev);
		}

		// select device
		CLDevice device = context.getMaxFlopsDevice();
		if (debug)
			System.out.println("Device: " + device);

		int imageSize = grid.getSize()[0] * grid.getSize()[1];
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxSIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ParallelProjectorRayDriven.cl"))
					.build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(imageSize, Mem.READ_ONLY);

		for (int i=0;i<grid.getSize()[1];++i){
			for (int j=0;j<grid.getSize()[0];++j)
				imageBuffer.getBuffer().put(((Grid2D)grid).getAtIndex(j, i));
		}

		imageBuffer.getBuffer().rewind();

		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(
				imageBuffer.getBuffer(), grid.getSize()[0], grid.getSize()[1],
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxSIndex, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven1DCL");
		kernel.putArg(imageGrid).putArg(sinogram)
		.putArg((float)maxS).putArg((float)deltaS)
		.putArg((float)maxTheta).putArg((float)deltaTheta)
		.putArg(maxSIndex).putArg(maxThetaIndex).putArg(index);

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteImage(imageGrid, true)
		.finish()
		.put1DRangeKernel(kernel, 0, globalWorkSizeT, localWorkSize).putBarrier()
		.putReadBuffer(sinogram, true)
		.finish();

		// write sinogram back to grid2D
		sinogram.getBuffer().rewind();

		for (int i = 0; i < ((Grid1D) sino).getBuffer().length; ++i) {
			((Grid1D) sino).setAtIndex(i, sinogram.getBuffer().get());
		}

		// clean up
		queue.release();
		imageGrid.release();
		sinogram.release();
		kernel.release();
		program.release();
		context.release();

		return sino;
	}
}


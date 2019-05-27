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

public class FanProjectorRayDrivenCL extends Projector{
	final double samplingRate = 3.d;

	private double focalLength;
	private double deltaBeta;
	private double maxT;
	private double deltaT;
	private double maxBeta;

	int maxTIndex, maxBetaIndex;

	/** 
	 * Creates a new instance of a fan-beam projector
	 * 
	 * @param focalLength the focal length
	 * @param maxBeta the maximal rotation angle
	 * @param deltaBeta the step size between source positions
	 * @param maxT the length of the detector array
	 * @param deltaT the size of one detector element
	 */
	public FanProjectorRayDrivenCL(double focalLength,double maxBeta, double deltaBeta,double maxT, double deltaT) {
		this.focalLength = focalLength;
		this.maxT = maxT;
		this.deltaBeta = deltaBeta;
		this.deltaT = deltaT;
		this.maxBetaIndex = (int) (maxBeta / deltaBeta + 1);
		this.maxTIndex = (int) (maxT / deltaT);
		this.maxBeta=maxBeta;
	}

	public NumericGrid project(NumericGrid grid, NumericGrid sino) {
		return this.projectRayDrivenCL((Grid2D)grid);
	}

	public NumericGrid project(NumericGrid grid, NumericGrid sino, int index) {
		return this.projectRayDriven1DCL((Grid2D) grid, index);
	}

	public Grid2D projectRayDrivenCL(Grid2D grid) {
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
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxTIndex); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeBeta = OpenCLUtil.roundUp(localWorkSize, maxBetaIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanProjectorRayDriven.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(imageSize, Mem.READ_ONLY);
		
		for (int i=0;i<grid.getBuffer().length;++i){
				imageBuffer.getBuffer().put(grid.getBuffer()[i]);
		}
		imageBuffer.getBuffer().rewind();
		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(
				imageBuffer.getBuffer(), grid.getSize()[0], grid.getSize()[1],
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxTIndex * maxBetaIndex, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven2DCL");
		kernel.putArg(imageGrid).putArg(sinogram)
			.putArg((float)maxT).putArg((float)deltaT)
			.putArg((float)maxBeta).putArg((float)deltaBeta)
			.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex); // TODO: Spacing :)

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
		Grid2D sino = new Grid2D(maxTIndex,maxBetaIndex);
		sino.setSpacing(deltaT, deltaBeta);
		sinogram.getBuffer().rewind();
		for (int i = 0; i < sino.getBuffer().length; ++i) {
				sino.getBuffer()[i] = sinogram.getBuffer().get();
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


	public Grid1D projectRayDriven1DCL(Grid2D grid, int index) {
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
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxTIndex); // rounded up to the nearest multiple of localWorkSize
		
		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanProjectorRayDriven.cl"))
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
				imageBuffer.getBuffer().put(grid.getAtIndex(j, i));
		}
		
		imageBuffer.getBuffer().rewind();
		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(
				imageBuffer.getBuffer(), grid.getSize()[0], grid.getSize()[1],
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxTIndex, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven1DCL");
		kernel.putArg(imageGrid).putArg(sinogram)
			.putArg((float)maxT).putArg((float)deltaT)
			.putArg((float)maxBeta).putArg((float)deltaBeta)
			.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex).putArg(index); // TODO: Spacing :)//*******

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
			.putWriteImage(imageGrid, true)
			.finish()
			.put1DRangeKernel(kernel, 0, globalWorkSizeT,localWorkSize).putBarrier()//**************************
			.putReadBuffer(sinogram, true)
			.finish();

		// write sinogram back to grid2D
		Grid1D sino = new Grid1D(new float [maxTIndex]);
		sino.setSpacing(deltaT);
		sinogram.getBuffer().rewind();
		for (int i = 0; i < sino.getBuffer().length; ++i) {
			sino.setAtIndex(i, sinogram.getBuffer().get());
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



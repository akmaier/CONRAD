package edu.stanford.rsl.science.iterativedesignstudy;

import java.io.IOException;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;


/**
 * 
 * @author Lina Felsner
 *
 */
public class ParallelBackprojectorRayDrivenCL extends Projector {

	final double samplingRate = 3.d;
	private int maxThetaIndex;
	private int maxSIndex; // image dimensions [GU]
	private double maxTheta;
	private double deltaTheta; // detector angles [rad]
	private double maxS;
	private double deltaS; // detector (pixel) size [mm]
	
	/**
	 * Sampling of projections is defined in the constructor.
	 */
	public ParallelBackprojectorRayDrivenCL(Grid2D sino) {
		this.maxThetaIndex = sino.getSize()[1];
		this.deltaTheta = sino.getSpacing()[1];
		this.maxTheta = (maxThetaIndex - 1) * deltaTheta;
		this.maxSIndex = sino.getSize()[0];
		this.deltaS = sino.getSpacing()[0];
		this.maxS = (maxSIndex - 1) * deltaS;
	}
	
	@Override
	public NumericGrid project(NumericGrid grid, NumericGrid sino) {
		NumericPointwiseOperators.fill(grid, 0);
		grid.setOrigin(-((grid.getSize()[0]-1) * grid.getSpacing()[0]) / 2.0,
				-((grid.getSize()[1]-1) * grid.getSpacing()[1]) / 2.0);
		
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

		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxSIndex); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeBeta = OpenCLUtil.roundUp(localWorkSize, maxThetaIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ParallelBackrojectorRayDriven.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input sino
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sino.getSize()[1], Mem.READ_ONLY);
		for (int i=0;i<sino.getSize()[1];++i){
				sinoBuffer.getBuffer().put(((Grid2D)sino).getBuffer()[i]);
		}
		sinoBuffer.getBuffer().rewind();
		
		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(sinoBuffer.getBuffer(), sino.getSize()[0], sino.getSize()[1], format);
		sinoBuffer.release();

		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(grid.getSize()[0] * grid.getSize()[1], Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("backprojectRayDriven1DCL"); // TODO implement ray driven backproijector opencl
		kernel.putArg(sinoGrid).putArg(imgBuffer)
		.putArg((float)maxS).putArg((float)deltaS)
		.putArg((float)maxTheta).putArg((float)deltaTheta)
		.putArg(maxSIndex).putArg(maxThetaIndex);
		
		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteImage(sinoGrid, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeBeta, globalWorkSizeT,
				localWorkSize, localWorkSize).putBarrier()
				.putReadBuffer(imgBuffer, true)
				.finish();

		imgBuffer.getBuffer().rewind();
		for (int j = 0; j < ((Grid2D) sino).getBuffer().length; ++j) {
			((Grid2D) sino).getBuffer()[j] = imgBuffer.getBuffer().get();
		}
		// createCommandQueue
		queue.putWriteImage(sinoGrid, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeT,
				globalWorkSizeBeta, localWorkSize, localWorkSize).finish()
				.putReadBuffer(imgBuffer, true).finish();

		imgBuffer.getBuffer().rewind();

		// write reco back to grid2D
		for (int i = 0; i < grid.getSize()[1]; ++i) {
			for (int j = 0; j < grid.getSize()[0]; ++j) {
				((Grid2D) grid).setAtIndex(i, j, imgBuffer.getBuffer().get());
			}
		}
		
		// clean up
		queue.release();
		sinoGrid.release();
		imgBuffer.release();
		kernel.release();
		program.release();
		context.release();

		float normalizationFactor = (float) ((float) samplingRate * maxThetaIndex / deltaS / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);
		return grid;
		
	}

	@Override
	public NumericGrid project(NumericGrid grid, NumericGrid sino, int index) {
		return null;
	}

}

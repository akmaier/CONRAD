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
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class ParallelBackprojectorPixelDrivenCL implements Backprojector {
	private int maxThetaIndex;
	private int maxSIndex; // image dimensions [GU]
	private double maxTheta;
	private double deltaTheta; // detector angles [rad]
	private double maxS;
	private double deltaS; // detector (pixel) size [mm]
	private int imgSizeX;
	private int imgSizeY; // image dimensions [GU]


	public ParallelBackprojectorPixelDrivenCL(NumericGrid sino, int imageSizeX, int imageSizeY) {
		this.maxThetaIndex = sino.getSize()[1];
		this.deltaTheta = sino.getSpacing()[1];
		this.maxTheta = (maxThetaIndex - 1) * deltaTheta;
		this.maxSIndex = sino.getSize()[0];
		this.deltaS = sino.getSpacing()[0];
		this.maxS = (maxSIndex - 1) * deltaS;
		this.imgSizeX = imageSizeX;
		this.imgSizeY = imageSizeY;
	}

	@Override
	public NumericGrid backproject(NumericGrid sino, NumericGrid grid) {
		//TODO Don't forget the Spacing >:)
		return null;
	}

	@Override
	public NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) {
		NumericPointwiseOperators.fill(grid, 0);
		grid.setOrigin(-((grid.getSize()[0]-1) * grid.getSpacing()[0]) / 2.0,
				-((grid.getSize()[1]-1) * grid.getSpacing()[1]) / 2.0);
		
		boolean debug = false;
		// create context
		CLContext context = OpenCLUtil.createContext();
		if (debug)
			System.out.println("Context: " + context);
		// show OpenCL devices in System
		CLDevice[] devices = context.getDevices();
		if (debug) {
			for (CLDevice dev : devices)
				System.out.println(dev);
		}

		// select device
		CLDevice device = context.getMaxFlopsDevice();
		if (debug)
			System.out.println("Device: " + device);

		// Length of arrays to process
		int sinoSize = projection.getNumberOfElements(); //TODO CHECK

		// localWorkSize
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8);
		// rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX);
		// rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY);

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(
					this.getClass().getResourceAsStream(
							"ParallelBackprojectorPixelDriven.cl")).build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}

		CLCommandQueue queue = device.createCommandQueue();
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY,
				ChannelType.FLOAT);
		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sinoSize,
				Mem.READ_ONLY);
		for (int i = 0; i < sinoSize; ++i) {
			sinoBuffer.getBuffer().put(((Grid1D) projection).getBuffer()[i]); //TODO CHECK
		}
		sinoBuffer.getBuffer().rewind();

		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
				sinoBuffer.getBuffer(), sinoSize, 1, format);

		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX
				* imgSizeY, Mem.READ_WRITE);

		// copy params
		CLKernel kernel = program
				.createCLKernel("backprojectPixelDriven2DOpenCL");
		kernel.putArg(sinoGrid).putArg(imgBuffer).putArg(imgSizeX)
		.putArg(imgSizeY).putArg((float) maxS).putArg((float) deltaS)
		.putArg((float) deltaTheta).putArg((float) grid.getSpacing()[0])
		.putArg((float) grid.getSpacing()[1]).putArg(maxSIndex)
		.putArg(index);

		// createCommandQueue
		queue.putWriteImage(sinoGrid, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX,
				globalWorkSizeY, localWorkSize, localWorkSize).finish()
				.putReadBuffer(imgBuffer, true).finish();

		imgBuffer.getBuffer().rewind();

		// write sinogram back to grid2D
		for (int i = 0; i < imgSizeY; ++i) {
			for (int j = 0; j < imgSizeX; ++j) {
				((Grid2D) grid).setAtIndex(i, j, imgBuffer.getBuffer().get());
			}
		}

		// clean up
		queue.release();
		imgBuffer.release();
		sinoBuffer.release();
		sinoGrid.release();
		kernel.release();
		program.release();
		context.release();

		// scaling
		NumericPointwiseOperators.divideBy(grid, (float) (maxThetaIndex / Math.PI));
		return grid;
	}
}

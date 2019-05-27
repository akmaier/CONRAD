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

public class FanBackprojectorPixelDrivenCL implements Backprojector{
	final double samplingRate = 3.d;

	private float focalLength;
	private float deltaT;
	private float deltaBeta;
	private float maxT;
	private float maxBeta;
	private int	maxTIndex;
	private int maxBetaIndex;
	private int imgSizeX;
	private int imgSizeY;

	public FanBackprojectorPixelDrivenCL(double focalLength, double deltaT, double deltaBeta ,int imageSizeX, int imageSizeY, double  maxT, int maxTIndex, double maxBeta, int maxBetaIndex) {
		this.focalLength = (float) focalLength;
		this.deltaBeta = (float) deltaBeta;
		this.deltaT = (float) deltaT;
		this.imgSizeX = imageSizeX;
		this.imgSizeY = imageSizeY;
		this.maxT = (float) maxT;
		this.maxTIndex = maxTIndex;
		this.maxBeta = (float) maxBeta;
		this.maxBetaIndex = maxBetaIndex;
	}	

	public NumericGrid backproject(NumericGrid sino, NumericGrid grid) {
		return this.backprojectPixelDrivenCL((Grid2D) sino, (Grid2D) grid);
	}

	public NumericGrid backproject(NumericGrid projection, NumericGrid grid, int index) {
		return this.backprojectPixelDriven1DCL((Grid1D) projection, (Grid2D) grid, index);
	}

	public Grid2D backprojectPixelDrivenCL(Grid2D sino, Grid2D grid) {
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

		int sinoSize = maxBetaIndex*maxTIndex;
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); // Local work size dimensions
		int globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBackprojectorPixelDriven.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sinoSize, Mem.READ_ONLY);
		for (int i=0;i<sinoSize;++i){
			sinoBuffer.getBuffer().put(((Grid2D) sino).getBuffer()[i]);
		}
		sinoBuffer.getBuffer().rewind();

		/// CP
		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
				sinoBuffer.getBuffer(), sino.getSize()[0], sino.getSize()[1],
				format);
		sinoBuffer.release();

		// create memory for output image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("backprojectPixelDriven2DCL");
		kernel.putArg(sinoGrid).putArg(imgBuffer)
		.putArg(imgSizeX).putArg(imgSizeY)
		.putArg((float)maxT).putArg((float)deltaT)
		.putArg((float)maxBeta).putArg((float)deltaBeta)
		.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex); // TODO: Spacing :)

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteImage(sinoGrid, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,
				localWorkSize, localWorkSize)
				.finish()
				.putReadBuffer(imgBuffer, true)
				.finish();

		// write grid back to grid2D
		Grid2D img = new Grid2D(this.imgSizeX, this.imgSizeY);
//		img.setSpacing(200, pxSzYMM);
		imgBuffer.getBuffer().rewind();
		for (int i = 0; i < imgSizeX*imgSizeY; ++i) {
			((Grid2D) img).getBuffer()[i] = imgBuffer.getBuffer().get();
		}
		queue.release();
		imgBuffer.release();
		sinoGrid.release();
		kernel.release();
		program.release();
		context.release();

		return img;
	}


	public Grid2D backprojectPixelDriven1DCL(Grid1D projection, Grid2D grid,int index){
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

		int sinoSize = projection.getNumberOfElements();
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); // Local work size dimensions
		int globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize
		//int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxTIndex); 
		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBackprojectorPixelDriven.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sinoSize, Mem.READ_ONLY);
		for (int i=0;i<sinoSize;++i){
				sinoBuffer.getBuffer().put(projection.getBuffer()[i]);
		}
		sinoBuffer.getBuffer().rewind();
		
		/// CP
		/*
		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
				sinoBuffer.getBuffer(), sino.getSize()[0], sino.getSize()[1],
				format);
				*/
		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
			sinoBuffer.getBuffer(), sinoSize, 1, 
				format);
		//sinoBuffer.release();

		// create memory for output image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("backprojectPixelDriven1DCL");
		kernel.putArg(sinoGrid).putArg(imgBuffer)
		.putArg(imgSizeX).putArg(imgSizeY)
		.putArg((float)maxT).putArg((float)deltaT)
		.putArg((float)maxBeta).putArg((float)deltaBeta)
		.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex).putArg(index); // TODO: Spacing :)

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteImage(sinoGrid, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,
				localWorkSize, localWorkSize)
		.finish()
		.putReadBuffer(imgBuffer, true)
		.finish();

		// write grid back to grid2D
		Grid2D img = new Grid2D(this.imgSizeX, this.imgSizeY);
		//img.setSpacing(pxSzXMM, pxSzYMM);
		imgBuffer.getBuffer().rewind();
		for (int i = 0; i < imgSizeX*imgSizeY; ++i) {
				((Grid2D) img).getBuffer()[i] = imgBuffer.getBuffer().get();
		}
		queue.release();
		imgBuffer.release();
		sinoGrid.release();
		kernel.release();
		program.release();
		context.release();

		return img;
	}

}

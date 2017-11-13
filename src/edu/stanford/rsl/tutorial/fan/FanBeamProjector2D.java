package edu.stanford.rsl.tutorial.fan;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;


public class FanBeamProjector2D{
	final double samplingRate = 3.d;

	private double focalLength, maxBeta, deltaBeta, maxT, deltaT;

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
	public FanBeamProjector2D(double focalLength,
			double maxBeta, double deltaBeta,
			double maxT, double deltaT) {

		this.focalLength = focalLength;
		this.maxBeta = maxBeta;
		this.maxT = maxT;
		this.deltaBeta = deltaBeta;
		this.deltaT = deltaT;
		//this.maxBetaIndex = (int) (maxBeta / deltaBeta + 1);
		this.maxBetaIndex = (int) (maxBeta / deltaBeta );
		this.maxTIndex = (int) (maxT / deltaT);
		
	}
	
	public Grid2D projectRayDriven(Grid2D grid) {
		Grid2D sino = new Grid2D(maxTIndex, maxBetaIndex);
		sino.setSpacing(deltaT, deltaBeta);

		
		// create translation to the grid origin
		Translation trans = new Translation(-grid.getSize()[0] / 2.0,
				-grid.getSize()[1] / 2.0, -1);
		// build the inverse translation
		Transform inverse = trans.inverse();
		
		// set up image bounding box and translate to origin
		Box b = new Box(grid.getSize()[0], grid.getSize()[1], 2);
		b.applyTransform(trans);

		// iterate over the rotation angle
		for (int i = 0; i < maxBetaIndex; i++) {
			// compute the current rotation angle and its sine and cosine
			double beta = deltaBeta * i;
			double cosBeta = Math.cos(beta);
			double sinBeta = Math.sin(beta);
//			System.out.println(beta / Math.PI * 180);
			// compute source position
			PointND a = new PointND(focalLength * cosBeta, focalLength
					* sinBeta, 0.d);
			// compute end point of detector
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f
					* cosBeta, 0.d);

			// create an unit vector that points along the detector
			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			dirDetector.normalizeL2();

			// iterate over the detector elements
			for (int t = 0; t < maxTIndex; t++) {
				// calculate current bin position
				// the detector elements' position are centered
				double stepsDirection = 0.5f * deltaT + t * deltaT;
				PointND p = new PointND(p0);
				p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));
				
				// create a straight line between detector bin and source
				StraightLine line = new StraightLine(a, p);
				
				// find the line's intersection with the box
				ArrayList<PointND> points = b.intersect(line);
				
				// if we have two intersections build the integral 
				// otherwise continue with the next bin
				if (2 != points.size()) {
					if (points.size() == 0) {
						line.getDirection().multiplyBy(-1.d);
						points = b.intersect(line);
						if (points.size() == 0)
							continue;
					} else {
						continue; // last possibility:
						 // a) it is only one intersection point (exactly one of the boundary vertices) or
						 // b) it are infinitely many intersection points (along one of the box boundaries).
						 // c) our code is wrong
					}
					
				}

				// Extract intersections
				PointND start = points.get(0);
				PointND end = points.get(1);

				// get the normalized increment
				SimpleVector increment = new SimpleVector(end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				double distance = increment.normL2();
				increment.divideBy(distance * samplingRate);

				double sum = .0;
				start = inverse.transform(start);

				double incrementLength = increment.normL2();
				
				// compute the integral along the line
				for (double tLine = 0.0; tLine < distance * samplingRate; ++tLine) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(tLine));
					if (grid.getSize()[0] <= current.get(0) + 1
							|| grid.getSize()[1] <= current.get(1) + 1
							|| current.get(0) < 0 || current.get(1) < 0)
						continue;
					
					sum += InterpolationOperators.interpolateLinear(grid,
							current.get(0), current.get(1));
				}

				// normalize by the number of interpolation points
				sum /= samplingRate;
				// write integral value into the sinogram.
				sino.setAtIndex(t, i, (float) sum);
			}
		}
		return sino;
	}
	public Grid1D projectRayDriven1D(Grid2D grid, int Betaindex) {		
		Grid1D sino=new Grid1D(this.maxTIndex);
		sino.setSpacing(this.deltaT);
		// create translation to the grid origin
		Translation trans = new Translation(-grid.getSize()[0] / 2.0,-grid.getSize()[1] / 2.0, -1);
		// build the inverse translation
		Transform inverse = trans.inverse();

		// set up image bounding box and translate to origin
		Box b = new Box(grid.getSize()[0], grid.getSize()[1], 2);
		b.applyTransform(trans);

		// compute the current rotation angle and its sine and cosine
		double beta = deltaBeta * Betaindex;
		double cosBeta = Math.cos(beta);
		double sinBeta = Math.sin(beta);

		// compute source position
		PointND a = new PointND(focalLength * cosBeta, focalLength
				* sinBeta, 0.d);
		// compute end point of detector
		PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f
				* cosBeta, 0.d);

		// create an unit vector that points along the detector
		SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
		dirDetector.normalizeL2();

		// iterate over the detector elements
		for (int t = 0; t < maxTIndex; t++) {
			// calculate current bin position
			// the detector elements' position are centered
			double stepsDirection = 0.5f * deltaT + t * deltaT;
			PointND p = new PointND(p0);
			p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));

			// create a straight line between detector bin and source
			StraightLine line = new StraightLine(a, p);

			// find the line's intersection with the box
			ArrayList<PointND> points = b.intersect(line);

			// if we have two intersections build the integral 
			// otherwise continue with the next bin
			if (2 != points.size()) {
				if (points.size() == 0) {
					line.getDirection().multiplyBy(-1.d);
					points = b.intersect(line);
					if (points.size() == 0)
						continue;
				} else {
					continue; // last possibility:
					// a) it is only one intersection point (exactly one of the boundary vertices) or
					// b) it are infinitely many intersection points (along one of the box boundaries).
					// c) our code is wrong
				}

			}

			// Extract intersections
			PointND start = points.get(0);
			PointND end = points.get(1);

			// get the normalized increment
			SimpleVector increment = new SimpleVector(end.getAbstractVector());
			increment.subtract(start.getAbstractVector());
			double distance = increment.normL2();
			increment.divideBy(distance * samplingRate);

			double sum = .0;
			start = inverse.transform(start);

			// compute the integral along the line
			for (double tLine = 0.0; tLine < distance * samplingRate; ++tLine) {
				PointND current = new PointND(start);
				current.getAbstractVector().add(increment.multipliedBy(tLine));
				if (grid.getSize()[0] <= current.get(0) + 1
						|| grid.getSize()[1] <= current.get(1) + 1
						|| current.get(0) < 0 || current.get(1) < 0)
					continue;

				sum += InterpolationOperators.interpolateLinear(grid,
						current.get(0), current.get(1));
			}

			// normalize by the number of interpolation points
			sum /= samplingRate;
			// write integral value into the sinogram.
			sino.setAtIndex(t, (float) sum);
		}
		return sino;
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
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamProjector.cl"))
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
				format,Mem.READ_ONLY);
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
	
	
	
	public Grid2D projectRayDrivenCL_moco(Grid2D grid) {
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
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamProjector_moco.cl"))
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
				format,Mem.READ_ONLY);
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
	
	
	
	
	
	
	
	
	
	
	
	
	public void fastProjectRayDrivenCL(OpenCLGrid2D sinoCL, OpenCLGrid2D gridCL) {

		sinoCL.getDelegate().prepareForDeviceOperation();
		gridCL.getDelegate().prepareForDeviceOperation();
		CLContext context = OpenCLUtil.getStaticContext();
		CLDevice device = context.getMaxFlopsDevice();

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxTIndex); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeBeta = OpenCLUtil.roundUp(localWorkSize, maxBetaIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamProjector.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		
		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(gridCL.getDelegate().getCLBuffer().getBuffer(), gridCL.getSize()[0], gridCL.getSize()[1],format,Mem.READ_ONLY);

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = sinoCL.getDelegate().getCLBuffer();

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven2DCL");
		kernel.putArg(imageGrid).putArg(sinogram)
			.putArg((float)maxT).putArg((float)deltaT)
			.putArg((float)maxBeta).putArg((float)deltaBeta)
			.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex); // TODO: Spacing :)

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
			.putCopyBufferToImage(gridCL.getDelegate().getCLBuffer(),imageGrid).finish()
			.put2DRangeKernel(kernel, 0, 0, globalWorkSizeBeta, globalWorkSizeT,localWorkSize, localWorkSize).putBarrier()
			.finish();
		
		kernel.rewind();
		sinoCL.getDelegate().notifyDeviceChange();
		sinoCL.setSpacing(deltaT, deltaBeta);
		// clean up
		queue.release();
		kernel.release();
		program.release();

	}
	
	public Grid1D projectRayDriven1DCL(Grid2D grid,int index) {
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
		//int globalWorkSizeBeta = OpenCLUtil.roundUp(localWorkSize, maxBetaIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamProjector.cl"))
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
				imageBuffer.getBuffer().put(grid.getAtIndex(j, i));
		}
		imageBuffer.getBuffer().rewind();
		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(
				imageBuffer.getBuffer(), grid.getSize()[0], grid.getSize()[1],
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxTIndex , Mem.WRITE_ONLY);//different from 2D openCL

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven1DCL");
		kernel.putArg(imageGrid).putArg(sinogram)
			.putArg((float)maxT).putArg((float)deltaT)
			.putArg((float)maxBeta).putArg((float)deltaBeta)
			.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex).putArg(index); // TODO: Spacing :)

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
			.putWriteImage(imageGrid, true)
			.finish()
			.put1DRangeKernel(kernel, 0, globalWorkSizeT,//***********************different from 2D opencl
					localWorkSize).putBarrier()
			.putReadBuffer(sinogram, true)
			.finish();

		// write sinogram back to grid2D
		Grid1D sino = new Grid1D(maxTIndex);
		sino.setSpacing(deltaT);
		sinogram.getBuffer().rewind();
		for (int i = 0; i < sino.getBuffer().length; ++i) {
				//sino.getBuffer()[i] = sinogram.getBuffer().get();
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

	public void fastProjectRayDrivenCL(OpenCLGrid1D sinoCL, OpenCLGrid2D gridCL,int index) {

		CLContext context = OpenCLUtil.getStaticContext();
		CLDevice device = context.getMaxFlopsDevice();

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxTIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamProjector.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLImage2d<FloatBuffer> imageGrid = context.createImage2d(gridCL.getDelegate().getCLBuffer().getBuffer(), gridCL.getSize()[0], gridCL.getSize()[1],format);

		// copy params
		CLKernel kernel = program.createCLKernel("projectRayDriven1DCL");
		kernel.putArg(imageGrid).putArg(sinoCL.getDelegate().getCLBuffer())
			.putArg((float)maxT).putArg((float)deltaT)
			.putArg((float)maxBeta).putArg((float)deltaBeta)
			.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex).putArg(index); 

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
			.putCopyBufferToImage(gridCL.getDelegate().getCLBuffer(),imageGrid).finish()
			.put1DRangeKernel(kernel, 0, globalWorkSizeT,localWorkSize).putBarrier()
			.finish();
		kernel.rewind();
		sinoCL.getDelegate().notifyDeviceChange();

		// clean up
		queue.release();
		imageGrid.release();
		kernel.release();
		program.release();

	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
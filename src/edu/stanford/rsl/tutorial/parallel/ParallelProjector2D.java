package edu.stanford.rsl.tutorial.parallel;

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
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;


/**
 * Implementation of a simple parallel projector. In order to create a
 * projection, the angular range and the angular sampling and the detector
 * element size and detector element sampling has to be defined. We show both, a
 * ray driven and a pixel driven projector.
 * 
 * See L. Zeng. "Medical Image Reconstruction: A Conceptual tutorial". 2009, page 3
 *
 * 
 * @author Recopra Seminar Summer 2012
 * 
 */
public class ParallelProjector2D {

	double maxTheta, deltaTheta,	// [rad]
		maxS, deltaS;				// [mm]
	int maxThetaIndex, maxSIndex;

	/**
	 * Sampling of projections is defined in the constructor.
	 * 
	 * @param maxTheta the angular range in radians
	 * @param deltaTheta the angular step size in radians
	 * @param maxS the detector size in [mm]
	 * @param deltaS the detector element size in [mm]
	 */
	public ParallelProjector2D(double maxTheta, double deltaTheta, double maxS,
			double deltaS) {
		this.maxS = maxS;
		this.maxTheta = maxTheta;
		this.deltaS = deltaS;
		this.deltaTheta = deltaTheta;
		this.maxSIndex = (int) (maxS / deltaS);
		this.maxThetaIndex = (int) (maxTheta / deltaTheta);
	}

	/**
	 * The ray driven solution.
	 * 
	 * @param grid the image
	 * @return the sinogram
	 */
	public Grid2D projectRayDriven(Grid2D grid) {
		final double samplingRate = 3.d; // # of samples per pixel
		Grid2D sino = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
		sino.setSpacing(deltaS, deltaTheta);

		// set up image bounding box in WC
		Translation trans = new Translation(
				-(grid.getSize()[0] * grid.getSpacing()[0])/2, -(grid.getSize()[1] * grid.getSpacing()[1])/2, -1
			);
		Transform inverse = trans.inverse();

		Box b = new Box((grid.getSize()[0] * grid.getSpacing()[0]), (grid.getSize()[1] * grid.getSpacing()[1]), 2);
		b.applyTransform(trans);

		for(int e=0; e<maxThetaIndex; ++e){
			// compute theta [rad] and angular functions.
			double theta = deltaTheta * e;
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);

			for (int i = 0; i < maxSIndex; ++i) {
				// compute s, the distance from the detector edge in WC [mm]
				double s = deltaS * i - maxS / 2;
				// compute two points on the line through s and theta
				// We use PointND for Points in 3D space and SimpleVector for directions.
				PointND p1 = new PointND(s * cosTheta, s * sinTheta, .0d);
				PointND p2 = new PointND(-sinTheta + (s * cosTheta),
						(s * sinTheta) + cosTheta, .0d);
				// set up line equation
				StraightLine line = new StraightLine(p1, p2);
				// compute intersections between bounding box and intersection line.
				ArrayList<PointND> points = b.intersect(line);

				// only if we have intersections
				if (2 != points.size()){
					if(points.size() == 0) {
						line.getDirection().multiplyBy(-1.d);
						points = b.intersect(line);
					}
					if(points.size() == 0)
						continue;
				}

				PointND start = points.get(0); // [mm]
				PointND end = points.get(1);   // [mm]

				// get the normalized increment
				SimpleVector increment = new SimpleVector(
						end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				double distance = increment.normL2();
				increment.divideBy(distance * samplingRate);

				double sum = .0;
				start = inverse.transform(start);

				// compute the integral along the line.
				for (double t = 0.0; t < distance * samplingRate; ++t) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(t));

					double x = current.get(0) / grid.getSpacing()[0],
							y = current.get(1) / grid.getSpacing()[1];

					if (grid.getSize()[0] <= x + 1
							|| grid.getSize()[1] <= y + 1
							|| x < 0 || y < 0)
						continue;

					sum += InterpolationOperators.interpolateLinear(grid, x, y);
				}

				// normalize by the number of interpolation points
				sum /= samplingRate;
				// write integral value into the sinogram.
				sino.setAtIndex(i, e, (float)sum);
			}
		}
		return sino;
	}

	public Grid2D projectRayDrivenCL(Grid2D grid) {
		boolean debug = true;
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
			program = context.createProgram(this.getClass().getResourceAsStream("ParallelProjector.cl"))
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
		Grid2D sino = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
		sino.setSpacing(deltaS, deltaTheta);
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

	/**
	 * The pixel driven solution.
	 * 
	 * @param grid the image
	 * @return the sinogram
	 */
	public Grid2D projectPixelDriven(Grid2D grid) {
		Grid2D sino = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
		sino.setSpacing(deltaS, deltaTheta);

		for (int i = 0; i < maxThetaIndex; i++) {
			double theta = deltaTheta * i;
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);

			SimpleVector dirDetector = new SimpleVector(cosTheta, sinTheta);

			// loop over all grid points
			// x,y are in the grid coordinate system
			// wx,wy are in the world coordinate system
			for (int x = 0; x < grid.getSize()[0]; x++) {
				for (int y = 0; y < grid.getSize()[1]; y++) {
					float val = (float) (grid.getAtIndex(x, y)/deltaS);
					val *= grid.getSpacing()[0] * grid.getSpacing()[1]; // assuming isometric pixels
					double[] w = grid.indexToPhysical(x, y);
					double wx = w[0], wy = w[1]; // convenience
					SimpleVector pixel = new SimpleVector(wx, wy);
					double s = SimpleOperators.multiplyInnerProd(pixel,
							dirDetector);
					s += maxS/2;
					s /= deltaS;
					
					Grid1D subgrid = sino.getSubGrid(i);
					if (subgrid.getSize()[0] <= s + 1
							||  s < 0)
						continue;

					InterpolationOperators
							.addInterpolateLinear(subgrid, s, val);
				}
			}
		}

		return sino;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
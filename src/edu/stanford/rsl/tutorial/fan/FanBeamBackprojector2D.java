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
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class FanBeamBackprojector2D {
	final double samplingRate = 3.d;

	private float focalLength,
	deltaT, deltaBeta,
	maxT, maxBeta;
	private int	maxTIndex, maxBetaIndex;
	private int imgSizeX, imgSizeY;

	public FanBeamBackprojector2D(double focalLength, double deltaT, double deltaBeta ,int imageSizeX, int imageSizeY) {
		this.focalLength = (float) focalLength;
		this.deltaBeta = (float) deltaBeta;
		this.deltaT = (float) deltaT;
		this.imgSizeX = imageSizeX;
		this.imgSizeY = imageSizeY;

	}

	private void initSinogramParams(Grid2D sino) {
		this.maxTIndex = sino.getSize()[0];
		this.deltaT = (float) sino.getSpacing()[0];
		this.maxT = maxTIndex * deltaT;
		
		this.maxBetaIndex = sino.getSize()[1];
		this.deltaBeta = (float) sino.getSpacing()[1];
		this.maxBeta = (maxBetaIndex -1) * deltaBeta;
		
		return;
	}

	public Grid2D backprojectRayDrivenCL(Grid2D sino) {
		this.initSinogramParams(sino);
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

		int sinoSize = maxBetaIndex*maxTIndex;
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); // Local work size dimensions
		int globalWorkSizeT = OpenCLUtil.roundUp(localWorkSize, maxTIndex); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeBeta = OpenCLUtil.roundUp(localWorkSize, maxBetaIndex); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamBackProjectorRay.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(sinoSize, Mem.READ_ONLY);
		for (int i=0;i<sino.getNumberOfElements();++i){
				sinoBuffer.getBuffer().put(sino.getBuffer()[i]);
		}
		sinoBuffer.getBuffer().rewind();

		// create memory for output image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY, Mem.WRITE_ONLY);

		// copy params
		CLKernel kernel = program.createCLKernel("backprojectRayDriven2DCL");
		kernel.putArg(sinoBuffer).putArg(imgBuffer)
		.putArg(imgSizeX).putArg(imgSizeY)
		.putArg((float)maxT).putArg((float)deltaT)
		.putArg((float)maxBeta).putArg((float)deltaBeta)
		.putArg((float)focalLength).putArg(maxTIndex).putArg(maxBetaIndex); // TODO: Spacing :)

		// createCommandQueue
		CLCommandQueue queue = device.createCommandQueue();
		queue
		.putWriteBuffer(sinoBuffer, true)
		.putWriteBuffer(imgBuffer, true)
		.finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeBeta, globalWorkSizeT,
				localWorkSize, localWorkSize).finish()
				.putReadBuffer(imgBuffer, true)
				.finish();

		// write grid back to grid2D
		Grid2D img = new Grid2D(this.imgSizeX, this.imgSizeY);
		//img.setSpacing(pxSzXMM, pxSzYMM);
		for (int i = 0; i < imgSizeX*imgSizeY; ++i) {
				img.getBuffer()[i] = imgBuffer.getBuffer().get();
		}
		
		queue.release();
		imgBuffer.release();
		sinoBuffer.release();
		kernel.release();
		program.release();
		context.release();

		return img;
	}

	/**
	 * Ray driven implementation of the Backprojector. This methods still contains a bug.
	 * ParkerWeighting does not work. Use Pixel-Driven instead.
	 * @param sino the sinogram for backprojection
	 * @return the backprojected image
	 */
	public Grid2D backprojectRayDriven(Grid2D sino) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);

		// set up image bounding box in WC
		Translation trans = new Translation(-grid.getSize()[0]/2.0, -grid.getSize()[1]/2.0, -1);
		// set up the inverse transform
		Transform inverse = trans.inverse();

		Box b = new Box(grid.getSize()[0], grid.getSize()[1], 2);
		b.applyTransform(trans);

		// iterate over the projection angles
		for(int e=0; e<maxBetaIndex; ++e){
			// compute beta [rad] and angular functions.
			float beta = (float) (deltaBeta * e);
			float cosBeta = (float) Math.cos(beta);
			float sinBeta = (float) Math.sin(beta);
			
			// We use PointND for points in 3D space and SimpleVector for directions.
			// compute source location and the beginning of the detector
			PointND a = new PointND(focalLength * cosBeta, focalLength * sinBeta, 0.d);
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta, 0.d);

			// compute the normalized vector along the detector
			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			dirDetector.normalizeL2();

			// iterate over the detector bins
			for (int i = 0; i < maxTIndex; ++i) {
				
				// compute bin position
				float stepsDirection = (float) (0.5f * deltaT + i * deltaT);
				PointND p = new PointND(p0);
				p.getAbstractVector().add(dirDetector.multipliedBy(stepsDirection));
				// set up line equation
				StraightLine line = new StraightLine(a, p);
				// compute intersections between bounding box and intersection line.
				ArrayList<PointND> points = b.intersect(line);

				// only if we have intersections
				if (2 != points.size()){
					if(points.size() == 0) {
						line.getDirection().multiplyBy(-1.f);
						points = b.intersect(line);
					}
					if(points.size() == 0)
						continue;
				}

				PointND start = points.get(0);
				PointND end = points.get(1);

				// get the normalized increment
				SimpleVector increment = new SimpleVector(end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				float distance = (float) increment.normL2();
				increment.divideBy(distance * samplingRate);

				
				float val = sino.getAtIndex(i, e);
				start = inverse.transform(start);

				// compute the integral along the line.
				for (float t = 0.0f; t < distance * samplingRate; ++t) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(t));
					if (grid.getSize()[0] <= current.get(0) + 1
							|| grid.getSize()[1] <= current.get(1) + 1
							|| current.get(0) < 0 || current.get(1) < 0)
						continue;

					//DistanceWeighting
					PointND currentTrans = new PointND(current);
					currentTrans.applyTransform(trans);
					float radius = (float) currentTrans.getAbstractVector().normL2();
					float phi = (float) (Math.PI/2 + Math.atan2(currentTrans.get(1), currentTrans.get(0)));
					float dWeight = (float) ((focalLength  + radius*Math.sin(beta - phi))/focalLength);
					float valtemp = (float) (val / (dWeight*dWeight));

					//grid.addAtIndex((int)Math.floor(current.get(0)), (int)Math.floor(current.get(1)), valtemp);
					InterpolationOperators.addInterpolateLinear(grid, current.get(1), current.get(0), valtemp);//FIXME
				}
			}
		}
		float normalizationFactor = (float) ((float) samplingRate * maxBetaIndex / deltaT / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);//FIXME

		return grid;
	}

	public Grid2D backprojectPixelDrivenCL(Grid2D sino) {
		this.initSinogramParams(sino);
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

		int sinoSize = maxBetaIndex*maxTIndex;
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); // Local work size dimensions
		int globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("FanBeamBackProjectorPixel.cl"))
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
				sinoBuffer.getBuffer().put(sino.getBuffer()[i]);
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
		//img.setSpacing(pxSzXMM, pxSzYMM);
		imgBuffer.getBuffer().rewind();
		for (int i = 0; i < imgSizeX*imgSizeY; ++i) {
				img.getBuffer()[i] = imgBuffer.getBuffer().get();
		}
		queue.release();
		imgBuffer.release();
		sinoGrid.release();
		kernel.release();
		program.release();
		context.release();

		return img;
	}

	/**
	 * The pixel driven solution for back-projection. 
	 * 
	 * @param sino
	 *            the sinogram
	 * @return the image
	 */
	public Grid2D backprojectPixelDriven(Grid2D sino) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);

		
		//ImageStack g  = new ImageStack(imgSizeX, imgSizeY, maxBetaIndex);
		//ImageStack g1  = new ImageStack(imgSizeX, imgSizeY, maxBetaIndex);
		//ImageStack g2  = new ImageStack(imgSizeX, imgSizeY, maxBetaIndex);
		
		for(int b=0; b<maxBetaIndex; ++b){
			//Grid2D gridIntermediate = new Grid2D(new float[this.imgSizeX][this.imgSizeY]);
			//Grid2D distWeights = new Grid2D(new float[this.imgSizeX][this.imgSizeY]);
			
			// compute beta [rad] and angular functions.
			float beta = (float) (deltaBeta * b);
			float cosBeta = (float) Math.cos(beta);
			float sinBeta = (float) Math.sin(beta);

			PointND a = new PointND(focalLength * cosBeta, focalLength * sinBeta, 0.d);
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta, 0.d);

			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			StraightLine detectorLine = new StraightLine(p0, dirDetector);

			Grid1D subSino = sino.getSubGrid(b);

			for(int x=0; x<imgSizeX; ++x){
				float wx = (x - imgSizeX/2.0f);

				for (int y = 0; y < imgSizeY; ++y) {
					float wy = (y - imgSizeY/2.0f);

					// compute two points on the line through t and beta
					// We use PointND for points in 3D space and SimpleVector for directions.
					final PointND reconstructionPointWorld = new PointND(wx, wy, 0.d);

					final StraightLine projectionLine = new StraightLine(a, reconstructionPointWorld);
					final PointND detectorPixel = projectionLine.intersect(detectorLine);
					
					float valtemp;
					if (detectorPixel != null)
					{
						final SimpleVector p = SimpleOperators.subtract(
							detectorPixel.getAbstractVector(), p0.getAbstractVector()
							);
						double len = p.normL2();
						if((p.getElement(0)*dirDetector.getElement(0)+p.getElement(1)*dirDetector.getElement(1))<0)
							len=-len;//Dealing with truncation data
						double t = (len-0.5d)/deltaT;

						if (subSino.getSize()[0] <= t + 1 ||  t < 0)
							continue;
						
						float val = InterpolationOperators.interpolateLinear(subSino, t);
						
						//DistanceWeighting
						float radius = (float) reconstructionPointWorld.getAbstractVector().normL2();
						float phi = (float) ((Math.PI/2) + Math.atan2(reconstructionPointWorld.get(1), reconstructionPointWorld.get(0)));
						float dWeight = (float) ((focalLength  +radius*Math.sin(beta - phi))/focalLength);
						valtemp = (float) (val / (dWeight*dWeight));
						
						//distWeights.setAtIndex(x, y, 1.0f/(dWeight*dWeight));
					}
					else
					{
						//final PointND detectorPixel2 = projectionLine.intersect(detectorLine);
						//if (detectorPixel2 == null) {}
						valtemp = 0.f;
						//distWeights.setAtIndex(x, y, 100.0f);
					}
					
					grid.addAtIndex(x, y, valtemp);
					//gridIntermediate.addAtIndex(x, y, valtemp);
				}
			}
			
/*			Grid2DProcessor g2dp = new Grid2DProcessor(gridIntermediate);
			g.setPixels(g2dp.getPixels(), b+1);
			g.setSliceLabel("Beta: " + Double.toString(beta*180/Math.PI), b+1);
			
			NumericalPointwiseOperators.add(grid, gridIntermediate, grid);
			*/
			/*Grid2D tempGrid = new Grid2D(grid);
			NumericalPointwiseOperators.multiplyBy(tempGrid, (float)((float)maxBetaIndex/(float)(b+1)));
			Grid2DProcessor g2dp2 = new Grid2DProcessor(tempGrid);
			g1.setPixels(g2dp2.getPixels(), b+1);
			g1.setSliceLabel("Beta: " + Double.toString(beta*180/Math.PI), b+1);
/*
			
			Grid2DProcessor g2dp3 = new Grid2DProcessor(distWeights);
			g2.setPixels(g2dp3.getPixels(), b+1);
			g2.setSliceLabel("Beta: " + Double.toString(beta*180/Math.PI), b+1);*/
			
			
			
		} // end for
		
/*		ImagePlus ip = new ImagePlus();
		ip.setStack(g);
		ip.show();
		*/
		/*ImagePlus ip2 = new ImagePlus();
		ip2.setStack(g1);
		ip2.show();
		/*
		ImagePlus ip3 = new ImagePlus();
		ip3.setStack(g2);
		ip3.show();*/
		
		float normalizationFactor = (float) (maxBetaIndex / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);

		return grid;
	}
	
	
	/**
	 * The pixel driven solution for back-projection in a given range and with a given resolution. 
	 * 
	 * @param sino
	 *            the sinogram
	 * @return the image
	 */
	public Grid2D backprojectPixelDriven_HighResRegion(Grid2D sino, float PixelScaling, double[] ROIcenter) {
		this.initSinogramParams(sino);
		Grid2D grid = new Grid2D(this.imgSizeX, this.imgSizeY);
		grid.setSpacing(PixelScaling,PixelScaling);
		
		//ImageStack g  = new ImageStack(imgSizeX, imgSizeY, maxBetaIndex);
		//ImageStack g1  = new ImageStack(imgSizeX, imgSizeY, maxBetaIndex);
		//ImageStack g2  = new ImageStack(imgSizeX, imgSizeY, maxBetaIndex);
		
		for(int b=0; b<maxBetaIndex; ++b){
			//Grid2D gridIntermediate = new Grid2D(new float[this.imgSizeX][this.imgSizeY]);
			//Grid2D distWeights = new Grid2D(new float[this.imgSizeX][this.imgSizeY]);
			
			// compute beta [rad] and angular functions.
			float beta = (float) (deltaBeta * b);
			float cosBeta = (float) Math.cos(beta);
			float sinBeta = (float) Math.sin(beta);

			PointND a = new PointND(focalLength * cosBeta, focalLength * sinBeta, 0.d);
			PointND p0 = new PointND(-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta, 0.d);

			SimpleVector dirDetector = p0.getAbstractVector().multipliedBy(-1);
			StraightLine detectorLine = new StraightLine(p0, dirDetector);

			Grid1D subSino = sino.getSubGrid(b);

			for(int x=0; x<imgSizeX; ++x){
				float wx = (float)(ROIcenter[0] + (x - imgSizeX/2.0f+0.5f)*PixelScaling);

				for (int y = 0; y < imgSizeY; ++y) {
					float wy = (float)(ROIcenter[1] + (y - imgSizeX/2.0f+0.5f)*PixelScaling);

					// compute two points on the line through t and beta
					// We use PointND for points in 3D space and SimpleVector for directions.
					final PointND reconstructionPointWorld = new PointND(wx, wy, 0.d);

					final StraightLine projectionLine = new StraightLine(a, reconstructionPointWorld);
					final PointND detectorPixel = projectionLine.intersect(detectorLine);
					
					float valtemp;
					if (detectorPixel != null)
					{
						final SimpleVector p = SimpleOperators.subtract(
							detectorPixel.getAbstractVector(), p0.getAbstractVector()
							);
						double len = p.normL2();
						double t = (len-0.5d)/deltaT;

						if (subSino.getSize()[0] <= t + 1 ||  t < 0)
							continue;
						
						float val = InterpolationOperators.interpolateLinear(subSino, t);
						
						//DistanceWeighting
						float radius = (float) reconstructionPointWorld.getAbstractVector().normL2();
						float phi = (float) ((Math.PI/2) + Math.atan2(reconstructionPointWorld.get(1), reconstructionPointWorld.get(0)));
						float dWeight = (float) ((focalLength  +radius*Math.sin(beta - phi))/focalLength);
						valtemp = (float) (val / (dWeight*dWeight));
						
						//distWeights.setAtIndex(x, y, 1.0f/(dWeight*dWeight));
					}
					else
					{
						//final PointND detectorPixel2 = projectionLine.intersect(detectorLine);
						//if (detectorPixel2 == null) {}
						valtemp = 0.f;
						//distWeights.setAtIndex(x, y, 100.0f);
					}
					
					grid.addAtIndex(x, y, valtemp);
					//gridIntermediate.addAtIndex(x, y, valtemp);
				}
			}
			
/*			Grid2DProcessor g2dp = new Grid2DProcessor(gridIntermediate);
			g.setPixels(g2dp.getPixels(), b+1);
			g.setSliceLabel("Beta: " + Double.toString(beta*180/Math.PI), b+1);
			
			NumericalPointwiseOperators.add(grid, gridIntermediate, grid);
			
			Grid2DProcessor g2dp2 = new Grid2DProcessor(grid);
			g1.setPixels(g2dp2.getPixels(), b+1);
			g1.setSliceLabel("Beta: " + Double.toString(beta*180/Math.PI), b+1);

			
			Grid2DProcessor g2dp3 = new Grid2DProcessor(distWeights);
			g2.setPixels(g2dp3.getPixels(), b+1);
			g2.setSliceLabel("Beta: " + Double.toString(beta*180/Math.PI), b+1);*/
			
			
			
		} // end for
		
/*		ImagePlus ip = new ImagePlus();
		ip.setStack(g);
		ip.show();
		
		ImagePlus ip2 = new ImagePlus();
		ip2.setStack(g1);
		ip2.show();
		
		ImagePlus ip3 = new ImagePlus();
		ip3.setStack(g2);
		ip3.show();*/
		
		float normalizationFactor = (float) (maxBetaIndex / Math.PI);
		NumericPointwiseOperators.divideBy(grid, normalizationFactor);

		return grid;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
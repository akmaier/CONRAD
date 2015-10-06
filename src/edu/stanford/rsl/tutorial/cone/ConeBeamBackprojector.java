package edu.stanford.rsl.tutorial.cone;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.opencl.TestOpenCL;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class ConeBeamBackprojector {

	final boolean debug = false;
	final boolean verbose = false;

	private Trajectory geometry;
	
	//image variables
	private int imgSizeX;
	private int imgSizeY;
	private int imgSizeZ;
	private Projection[] projMats;
	private int maxProjs;
	private double spacingX;
	private double spacingY;
	private double spacingZ;
	private double originX;
	private double originY;
	private double originZ;
	
	//cl variables
	private CLContext context;
	private CLDevice device;
	private CLBuffer<FloatBuffer> projMatrices;
	private CLCommandQueue queue;
	private CLKernel kernel;
	// Length of arrays to process
	private int localWorkSize;
	private int globalWorkSizeX; 
	private int globalWorkSizeY; 
	private CLImageFormat format;
	private CLProgram program;
	
	//normalization parameter
	float normalizer;

	public ConeBeamBackprojector() {
		configure();
		initCL();
	}

	public void configure(){
		geometry = Configuration.getGlobalConfiguration().getGeometry();
		imgSizeX = geometry.getReconDimensionX();
		imgSizeY = geometry.getReconDimensionY();
		imgSizeZ = geometry.getReconDimensionZ();
		projMats = geometry.getProjectionMatrices();
		maxProjs = geometry.getProjectionStackSize();
		spacingX = geometry.getVoxelSpacingX();
		spacingY = geometry.getVoxelSpacingY();
		spacingZ = geometry.getVoxelSpacingZ();
		originX = -geometry.getOriginX();
		originY = -geometry.getOriginY();
		originZ = -geometry.getOriginZ();
		normalizer = (float) (geometry.getSourceToDetectorDistance()*geometry.getSourceToAxisDistance()*Math.PI / maxProjs);
	}
	
	private void initCL(){
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();

		// load sources, create and build program
		program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ConeBeamBackProjector.cl")).build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}
		kernel =  program.createCLKernel("backProjectPixelDrivenCL");
		
		// create image from input grid
		format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		
		localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16);
		globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); 
		globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); 
		
		projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		for(int p = 0; p < maxProjs; p++) {
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
		projMatrices.getBuffer().rewind();
		queue.putWriteBuffer(projMatrices, true).finish();
		
	}
	
	public void unload(){
		if(program != null && !program.isReleased())
			program.release();
		if(projMatrices != null && !projMatrices.isReleased())
			projMatrices.release();
	}

	public Grid3D backprojectPixelDriven(Grid2D sino, int projIdx) {

		configure();
		
		if(projIdx >= maxProjs || 0 > projIdx){
			System.err.println("ConeBeamBackprojector: Invalid projection index");
			return null;
		}

		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		grid.setOrigin(-originX, -originY, -originZ);
		grid.setSpacing(spacingX, spacingY, spacingZ);
		for(int x = 0; x < imgSizeX; x++) {
			double xTrans = x*spacingX-originX;
			for(int y = 0; y < imgSizeY ; y++) {
				double yTrans = y*spacingY-originY;
				for(int z = 0; z < imgSizeZ; z++) {
					SimpleVector point3d = new SimpleVector(xTrans, yTrans, z*spacingZ-originZ, 1);
					//for(int p = 0; p < maxProjs; p++) {
					int p = projIdx;
					SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
					double coordU = point2d.getElement(0) / point2d.getElement(2);
					double coordV = point2d.getElement(1) / point2d.getElement(2);

					float val = (float) (InterpolationOperators.interpolateLinear(sino, coordU, coordV)/(point2d.getElement(2)*point2d.getElement(2))); //
					//if(Float.isInfinite(val) || Float.isNaN(val))
					//	val = 0;
					grid.setAtIndex(x, y, z, val); // "set" instead of "add", because #proj==1
					//}
				}
			}
		}

		return grid;
	}

	public Grid3D backprojectPixelDriven(final Grid3D sino) {
		configure();
			
		final Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);

		int nThreads = Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS));
		// TODO Error-Checking for thread number
		ExecutorService executorService = Executors.newFixedThreadPool(nThreads);
		for(int i = 0; i < imgSizeX; i++) {
			final int x = i;
			executorService.execute(new Runnable(){
				@Override
				public void run(){
					System.out.println("Working on slice "+String.valueOf(x+1)+" of "+imgSizeX);
					double xTrans = x*spacingX-originX;
					for(int y = 0; y < imgSizeY ; y++) {
						double yTrans = y*spacingY-originY;
						for(int z = 0; z < imgSizeZ; z++) {
							SimpleVector point3d = new SimpleVector(xTrans, yTrans, z*spacingZ-originZ, 1);
							for(int p = 0; p < maxProjs; p++) {
								SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
								double coordU = point2d.getElement(0) / point2d.getElement(2);
								double coordV = point2d.getElement(1) / point2d.getElement(2);

								/*
								// TEST // TODO
								if(0==p)
									System.out.println("Sino - Min: " + sino.getGridOperator().min(sino) + " Max: " + sino.getGridOperator().max(sino));
								int uplusOne = Math.min((int)coordU+1, geo.getDetectorWidth()-1);
								int vplusOne = Math.min((int)coordV+1, geo.getDetectorHeight()-1);
								float[] values = new float[]{sino.getAtIndex((int)coordU, (int)coordV, p), sino.getAtIndex(uplusOne, (int)coordV, p),
										sino.getAtIndex((int)coordU, vplusOne, p), sino.getAtIndex(uplusOne, vplusOne, p)};
								float sumVal = values[0] + values[1] + values[2] + values[3];
								if (sumVal != 0)
									System.out.println("NOT NULL");
								// /TEST
								 */

								float val = (float) (InterpolationOperators.interpolateLinear(sino, p, coordU, coordV)/(point2d.getElement(2)*point2d.getElement(2)));

								/*
								// TEST // TODO
								if(sumVal/4 != val || 0 != val)
									System.out.println("[" + x + ", " + y + ", " +z + "] = " + val + " = interp(" + p + ", " + coordU + ", " + coordV + "), " + sumVal/4);
								// /TEST
								 */
								grid.addAtIndex(x, y, z, val);
							}
						}
					}
				}});
		}
		executorService.shutdown();
		try {
			executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			System.out.println("Exception waiting for thread termination: "+e);
		}

		return grid;
	}

	public void backprojectPixelDrivenCL(OpenCLGrid3D volume, OpenCLGrid2D[] sino) {

		for(int p = 0; p < maxProjs; p++) {

			CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(sino[p].getDelegate().getCLBuffer().getBuffer(), sino[p].getSize()[0], sino[p].getSize()[1],format,Mem.READ_ONLY);

			kernel.putArg(sinoGrid)
			.putArg(volume.getDelegate().getCLBuffer())
			.putArg(projMatrices)
			.putArg(p)
			.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
			.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
			.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ)
			.putArg(normalizer); 

			queue
			.putCopyBufferToImage(sino[p].getDelegate().getCLBuffer(), sinoGrid).finish()
			.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,localWorkSize, localWorkSize)
			.finish();
			
			kernel.rewind();
			sinoGrid.release();
		}

		volume.getDelegate().notifyDeviceChange();
	}

	public Grid3D backprojectPixelDrivenCL(Grid3D sino) {
		configure();
		
		OpenCLGrid2D [] sinoCL = new OpenCLGrid2D[sino.getSize()[2]];
		
		for (int i=0; i < sinoCL.length; i++){ sinoCL[i] = new OpenCLGrid2D(sino.getSubGrid(i)); sinoCL[i].getDelegate().prepareForDeviceOperation();}

		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		OpenCLGrid3D gridCL = new OpenCLGrid3D(grid);
		gridCL.getDelegate().prepareForDeviceOperation();

		backprojectPixelDrivenCL(gridCL, sinoCL);
		gridCL.setOrigin(-originX, -originY, -originZ);
		gridCL.setSpacing(spacingX, spacingY, spacingZ);
		for (int i=0; i < sinoCL.length; i++) sinoCL[i].release();
		grid = new Grid3D(gridCL);
		gridCL.release();
		unload();
		return grid;
	}

	public Grid3D backprojectPixelDrivenCL(Grid2D sino , int projIdx) {

		configure();

		OpenCLGrid2D sinoCL = new OpenCLGrid2D(sino);

		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		OpenCLGrid3D gridCL = new OpenCLGrid3D(grid);
		gridCL.getDelegate().prepareForDeviceOperation();

		backprojectPixelDrivenCL(gridCL, sinoCL, projIdx);

		gridCL.setOrigin(-originX, -originY, -originZ);
		gridCL.setSpacing(spacingX, spacingY, spacingZ);

		sinoCL.release();
		grid = new Grid3D(gridCL);
		gridCL.release();
		unload();
		return grid;
	}

	public void backprojectPixelDrivenCL(OpenCLGrid3D volume, OpenCLGrid2D sino, int projIdx) {

		//TODO MOEGLICHE FEHLERQUELLE
		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(sino.getDelegate().getCLBuffer().getBuffer(), sino.getSize()[0], sino.getSize()[1],format,Mem.READ_ONLY);

		kernel.putArg(sinoGrid)
		.putArg(volume.getDelegate().getCLBuffer())
		.putArg(projMatrices)
		.putArg(projIdx)
		.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
		.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
		.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ)
		.putArg(normalizer); 

		queue
		.putCopyBufferToImage(sino.getDelegate().getCLBuffer(), sinoGrid).finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,localWorkSize, localWorkSize)
		.finish();

		kernel.rewind();
		
		volume.getDelegate().notifyDeviceChange();
		sinoGrid.release();

	}

	public void fastBackprojectPixelDrivenCL(OpenCLGrid2D sinoCL, OpenCLGrid3D gridCL, int projIdx) {
		configure();
		
		gridCL.getDelegate().prepareForDeviceOperation();
		sinoCL.getDelegate().prepareForDeviceOperation();

		backprojectPixelDrivenCL(gridCL, sinoCL, projIdx);

	}
	
	public void fastBackprojectPixelDrivenCL(OpenCLGrid3D sinoCL, OpenCLGrid3D gridCL) {

		configure();

		gridCL.getDelegate().prepareForDeviceOperation();
		sinoCL.getDelegate().prepareForDeviceOperation();
		OpenCLGrid2D[] sinoBuf = new OpenCLGrid2D[sinoCL.getSize()[2]];
		for(int i = 0; i< sinoCL.getSize()[2];i++){
			sinoBuf[i] = new OpenCLGrid2D(sinoCL.getSubGrid(i));
			sinoBuf[i].getDelegate().prepareForDeviceOperation();
		}
		backprojectPixelDrivenCL(gridCL,sinoBuf);
		
		for(int i = 0; i< sinoCL.getSize()[2];i++)
			sinoBuf[i].release();
		unload();
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
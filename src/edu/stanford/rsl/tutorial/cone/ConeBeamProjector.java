package edu.stanford.rsl.tutorial.cone;

import java.io.IOException;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage3d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;



public class ConeBeamProjector {
	
	final boolean debug = false;
	final boolean verbose = false;

	public ConeBeamProjector() {
		Configuration.loadConfiguration();
	}
	
	public int getMaxProjections(){
		Configuration conf = Configuration.getGlobalConfiguration();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		return maxProjs;
	}
	
	public Grid2D projectPixelDriven(Grid3D grid, int projIdx) {
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int maxV = geo.getDetectorHeight();
		int maxU = geo.getDetectorWidth();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		if(projIdx+1 > maxProjs || 0 > projIdx){
			System.err.println("ConeBeamProjector: Invalid projection index");
			return null;
		}
		Grid2D sino = new Grid2D(maxU,maxV); //
		double spacingX = geo.getVoxelSpacingX();
		double spacingY = geo.getVoxelSpacingY();
		double spacingZ = geo.getVoxelSpacingZ();
		double originX = -geo.getOriginX();
		double originY = -geo.getOriginY();
		double originZ = -geo.getOriginZ();
		for (int x = 0; x < imgSizeX - 1; x++) {
			double xTrans = x * spacingX - originX;
			for (int y = 0; y < imgSizeY - 1; y++) {
				double yTrans = y * spacingY - originY;
				for (int z = 0; z < imgSizeZ - 1; z++) {
					SimpleVector point3d = new SimpleVector(xTrans, yTrans, z * spacingZ - originZ, 1);
					// for(int p = 0; p < maxProjs; p++) {
					int p = projIdx; //
					SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
					double coordU = point2d.getElement(0) / point2d.getElement(2);
					double coordV = point2d.getElement(1) / point2d.getElement(2);
					if (coordU >= maxU - 1 || coordV >= maxV - 1 || coordU <= 0 || coordV <= 0)
						continue;
					float val = grid.getAtIndex(x, y, z);
					InterpolationOperators.addInterpolateLinear(sino, coordU, coordV, val); //
					// }
				}
			}
		}

		return sino;
	}


	public Grid3D projectPixelDriven(Grid3D grid) {
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int maxV = geo.getDetectorHeight();
		int maxU = geo.getDetectorWidth();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		Grid3D sino = new Grid3D(maxU,maxV,maxProjs);
		double spacingX = geo.getVoxelSpacingX();
		double spacingY = geo.getVoxelSpacingY();
		double spacingZ = geo.getVoxelSpacingZ();
		double originX = -geo.getOriginX();
		double originY = -geo.getOriginY();
		double originZ = -geo.getOriginZ();
		for(int x = 0; x < imgSizeX-1; x++) {
			double xTrans = x*spacingX-originX;
			for(int y = 0; y < imgSizeY-1 ; y++) {
				double yTrans = y*spacingY-originY;
				for(int z = 0; z < imgSizeZ-1; z++) {
					SimpleVector point3d = new SimpleVector(xTrans, yTrans, z*spacingZ-originZ, 1);
					for(int p = 0; p < maxProjs; p++) {
						SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
						double coordU = point2d.getElement(0) / point2d.getElement(2);
						double coordV = point2d.getElement(1) / point2d.getElement(2);
						if (coordU >= maxU-1 || coordV>= maxV -1 || coordU <=0 || coordV <= 0)
							continue;
						float val = grid.getAtIndex(x, y, z);
						InterpolationOperators.addInterpolateLinear(sino, coordU, coordV, p, val);
					}
				}
			}
		}

		return sino;
	}
	
	public Grid2D projectRayDrivenCL(Grid3D grid, int projIdx) {
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int maxV = geo.getDetectorHeight();
		int maxU = geo.getDetectorWidth();
		float spacingU = (float) geo.getPixelDimensionX();
		float spacingV = (float) geo.getPixelDimensionY();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		if(projIdx+1 > maxProjs || 0 > projIdx){
			System.err.println("ConeBeamProjector: Invalid projection index");
			return null;
		}
		Grid2D sino = new Grid2D(maxV,maxU); //
		float spacingX = (float) geo.getVoxelSpacingX();
		float spacingY = (float) geo.getVoxelSpacingY();
		float spacingZ = (float) geo.getVoxelSpacingZ();
		float originX = (float) -geo.getOriginX();
		float originY = (float) -geo.getOriginY();
		float originZ = (float) -geo.getOriginZ();
		
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

		int imageSize = imgSizeX*imgSizeY*imgSizeZ;
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeU = OpenCLUtil.roundUp(localWorkSize, maxU); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeV = OpenCLUtil.roundUp(localWorkSize, maxV); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ConeBeamProjector.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(imageSize, Mem.READ_ONLY);
		for (int i=0;i<imgSizeZ;++i) {
			for (int j=0;j<imgSizeY;++j) {
				for(int k = 0; k < imgSizeX; k++)			
					imageBuffer.getBuffer().put(grid.getAtIndex(k, j, i));
			}
		}
		imageBuffer.getBuffer().rewind();
		CLImage3d<FloatBuffer> imageGrid = context.createImage3d(
				imageBuffer.getBuffer(), imgSizeX, imgSizeY, imgSizeZ,
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxU * maxV , Mem.WRITE_ONLY);

//		CLBuffer<floatBuffer> projMatrices = context.createfloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
//		for(int p = 0; p < maxProjs; p++) {
//			for(int row = 0; row < 3; row++) {
//				for(int col = 0; col < 4; col++) {
//					projMatrices.getBuffer().put(projMats[p].computeP().getElement(row, col));
//				}
//			}
//		}
		CLCommandQueue queue = device.createCommandQueue();
		queue
			//.putWriteBuffer(projMatrices, true)
			.putWriteImage(imageGrid, true).finish();
		// copy params
		CLKernel kernel =  program.createCLKernel("projectRayDrivenCL");
		//for(int p = 0; p < maxProjs; p++) {
		int p = projIdx; //
			
			SimpleVector source = projMats[p].computeCameraCenter();
			SimpleVector pAxis = projMats[p].computePrincipalAxis();
			kernel.putArg(imageGrid).putArg(sinogram)//.putArg(projMatrices)
			.putArg(p)
			.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
			.putArg(originX).putArg(originY).putArg(originZ)
			.putArg(spacingX).putArg(spacingY).putArg(spacingZ)
			.putArg(maxU).putArg(maxV)
			.putArg(spacingU).putArg(spacingV)
			.putArg((float)source.getElement(0)).putArg((float)source.getElement(1)).putArg((float)source.getElement(2))
			.putArg((float)pAxis.getElement(0)).putArg((float)pAxis.getElement(1)).putArg((float)pAxis.getElement(2)); 

			queue
			.put2DRangeKernel(kernel, 0, 0, globalWorkSizeU, globalWorkSizeV,
					localWorkSize, localWorkSize).putBarrier()
					.putReadBuffer(sinogram, true)
					.finish();
			
			sinogram.getBuffer().rewind();
			for (int v=0;v < maxV;++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u< maxU; u++){
					sino.setAtIndex(v,u, sinogram.getBuffer().get());
				}
			}
			sinogram.getBuffer().rewind();
			kernel.rewind();
		//}
		if (debug || verbose)
			System.out.println("Projection done!");
		// clean up
		imageGrid.release();
		sinogram.release();
		queue.release();
//		projMatrices.release();
		kernel.release();
		program.release();
		context.release();
	
		return sino;
	}

	public Grid3D projectRayDrivenCL(Grid3D grid) {
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int maxV = geo.getDetectorHeight();
		int maxU = geo.getDetectorWidth();
		float spacingU = (float) geo.getPixelDimensionX();
		float spacingV = (float) geo.getPixelDimensionY();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		Grid3D sino = new Grid3D(maxU,maxV,maxProjs);
		float spacingX = (float) geo.getVoxelSpacingX();
		float spacingY = (float) geo.getVoxelSpacingY();
		float spacingZ = (float) geo.getVoxelSpacingZ();
		float originX = (float) -geo.getOriginX();
		float originY = (float) -geo.getOriginY();
		float originZ = (float) -geo.getOriginZ();

		
		// create context
		CLContext context = OpenCLUtil.createContext();
		if (debug){
			System.out.println("Context: " + context);
			//show OpenCL devices in System
			CLDevice[] devices = context.getDevices();
			for (CLDevice dev: devices)
				System.out.println(dev);
		}

		// select device
		CLDevice device = context.getMaxFlopsDevice();
		if (debug)
			System.out.println("Device: " + device);

		int imageSize = imgSizeX*imgSizeY*imgSizeZ;
		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeU = OpenCLUtil.roundUp(localWorkSize, maxU); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeV = OpenCLUtil.roundUp(localWorkSize, maxV); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ConeBeamProjector.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		CLBuffer<FloatBuffer> imageBuffer = context.createFloatBuffer(imageSize, Mem.READ_ONLY);
		for (int i=0;i<imgSizeZ;++i) {
			for (int j=0;j<imgSizeY;++j) {
				for(int k = 0; k < imgSizeX; k++)			
					imageBuffer.getBuffer().put(grid.getAtIndex(k, j, i));
			}
		}
		imageBuffer.getBuffer().rewind();
		CLImage3d<FloatBuffer> imageGrid = context.createImage3d(
				imageBuffer.getBuffer(), imgSizeX, imgSizeY, imgSizeZ,
				format);
		imageBuffer.release();

		// create memory for sinogram
		CLBuffer<FloatBuffer> sinogram = context.createFloatBuffer(maxU * maxV , Mem.WRITE_ONLY);

//		CLBuffer<floatBuffer> projMatrices = context.createfloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
//		for(int p = 0; p < maxProjs; p++) {
//			for(int row = 0; row < 3; row++) {
//				for(int col = 0; col < 4; col++) {
//					projMatrices.getBuffer().put(projMats[p].computeP().getElement(row, col));
//				}
//			}
//		}
		CLCommandQueue queue = device.createCommandQueue();
		queue
			//.putWriteBuffer(projMatrices, true)
			.putWriteImage(imageGrid, true).finish();
		// copy params
		CLKernel kernel =  program.createCLKernel("projectRayDrivenCL");
		for(int p = 0; p < maxProjs; p++) {
			
			SimpleVector source = projMats[p].computeCameraCenter();
			SimpleVector pAxis = projMats[p].computePrincipalAxis();
			kernel.putArg(imageGrid).putArg(sinogram)//.putArg(projMatrices)
			.putArg(p)
			.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
			.putArg(originX).putArg(originY).putArg(originZ)
			.putArg(spacingX).putArg(spacingY).putArg(spacingZ)
			.putArg(maxU).putArg(maxV)
			.putArg(spacingU).putArg(spacingV)
			.putArg((float)source.getElement(0)).putArg((float)source.getElement(1)).putArg((float)source.getElement(2))
			.putArg((float)pAxis.getElement(0)).putArg((float)pAxis.getElement(1)).putArg((float)pAxis.getElement(2)); 

			queue
			.put2DRangeKernel(kernel, 0, 0, globalWorkSizeU, globalWorkSizeV,
					localWorkSize, localWorkSize).putBarrier()
					.putReadBuffer(sinogram, true)
					.finish();
			
			sinogram.getBuffer().rewind();
			for (int v=0;v < maxV;++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u< maxU; u++){
					sino.setAtIndex(u,v,p, sinogram.getBuffer().get());
				}
			}
			sinogram.getBuffer().rewind();
			kernel.rewind();
		}
		if (debug || verbose)
			System.out.println("Projection done!");
		// clean up
		imageGrid.release();
		sinogram.release();
		queue.release();
//		projMatrices.release();
		kernel.release();
		program.release();
		context.release();
	
		return sino;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
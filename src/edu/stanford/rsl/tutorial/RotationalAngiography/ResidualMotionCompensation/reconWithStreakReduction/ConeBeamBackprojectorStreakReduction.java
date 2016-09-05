package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.reconWithStreakReduction;


import java.io.IOException;
import java.nio.FloatBuffer;
import java.text.DecimalFormat;
import java.text.NumberFormat;

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
//import com.sun.tools.javac.resources.javac;




import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;

public class ConeBeamBackprojectorStreakReduction {
	
	private int gat_ign = 3;


	public ConeBeamBackprojectorStreakReduction() {

		Configuration.loadConfiguration();

	}

	public Grid3D backprojectPixelDriven(Grid2D sino, int projIdx) {

		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		if(projIdx+1 > maxProjs || 0 > projIdx){
			System.err.println("ConeBeamBackprojector: Invalid projection index");
			return null;
		}
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		double spacingX = geo.getVoxelSpacingX();
		double spacingY = geo.getVoxelSpacingY();
		double spacingZ = geo.getVoxelSpacingZ();
		double originX = -geo.getOriginX();
		double originY = -geo.getOriginY();
		double originZ = -geo.getOriginZ();
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

						float val = (float) (InterpolationOperators.interpolateLinear(sino, coordV, coordU)/(point2d.getElement(2)*point2d.getElement(2))); //

						grid.addAtIndex(x, y, z, val);
					//}
				}
			}
		}

		return grid;
	}


	public Grid3D backprojectPixelDriven(Grid3D sino) {

		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		double spacingX = geo.getVoxelSpacingX();
		double spacingY = geo.getVoxelSpacingY();
		double spacingZ = geo.getVoxelSpacingZ();
		double originX = -geo.getOriginX();
		double originY = -geo.getOriginY();
		double originZ = -geo.getOriginZ();
		for(int x = 0; x < imgSizeX; x++) {
			double xTrans = x*spacingX-originX;
			for(int y = 0; y < imgSizeY ; y++) {
				double yTrans = y*spacingY-originY;
				for(int z = 0; z < imgSizeZ; z++) {
					SimpleVector point3d = new SimpleVector(xTrans, yTrans, z*spacingZ-originZ, 1);
					for(int p = 0; p < maxProjs; p++) {
						SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
						double coordU = point2d.getElement(0) / point2d.getElement(2);
						double coordV = point2d.getElement(1) / point2d.getElement(2);

						float val = (float) (InterpolationOperators.interpolateLinear(sino, coordU, coordV, p)/(point2d.getElement(2)*point2d.getElement(2)));

						grid.addAtIndex(x, y, z, val);
					}
				}
			}
		}

		return grid;
	}

	public Grid3D backprojectPixelDrivenCL(Grid3D sino) {
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		int maxV = geo.getDetectorHeight();
		int maxU = geo.getDetectorWidth();
		int imgSizeX = geo.getReconDimensionX();
		int imgSizeY = geo.getReconDimensionY();
		int imgSizeZ = geo.getReconDimensionZ();
		Projection[] projMats = conf.getGeometry().getProjectionMatrices();
		int maxProjs = conf.getGeometry().getProjectionStackSize();
		
		
		//set maxProjs to BufferSize since Projections with a weight below a threshold were removed from the buffer to save computationtime
		maxProjs = sino.getBuffer().size();
		
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		double spacingX = geo.getVoxelSpacingX();
		double spacingY = geo.getVoxelSpacingY();
		double spacingZ = geo.getVoxelSpacingZ();
		double originX = -geo.getOriginX();
		double originY = -geo.getOriginY();
		double originZ = -geo.getOriginZ();

		boolean debug = true;
		if (debug)
			System.out.println("Backprojecting...");
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
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeZ = OpenCLUtil.roundUp(localWorkSize, imgSizeZ); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ConeBeamBackprojectorStreakReduction.cl"))
			//program = context.createProgram(this.getClass().getResourceAsStream("ConeBeamBackProjectorScatterCorrIterateOverY.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		/*
		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxProjs*maxU*maxV, Mem.READ_ONLY);
		//		for (int i = 0; i < grid.getSize()[0]; ++i) {
		//			imageBuffer.getBuffer().put(grid.getSubGrid(i).getBuffer());
		//		}

		for (int p=0;p<maxProjs;++p){
			for (int v=0;v<maxV;++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u < maxU; u++) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(p,v,u));
				}
			}
		}
		sinoBuffer.getBuffer().rewind();
		CLImage3d<FloatBuffer> sinoGrid = context.createImage3d(
				sinoBuffer.getBuffer(), maxU, maxV, maxProjs,	//TODO MOEGLICHE FEHLERQULEL
				format);
		sinoBuffer.release();
		*/

		// create memory for image
		CLBuffer<FloatBuffer> destBuffer = context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		//destBuffer.getBuffer().rewind();
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ *(2*gat_ign+1), Mem.WRITE_ONLY);
		int imgBufferSize = (2*gat_ign+1)*imgSizeX*imgSizeY*imgSizeZ;
		//not sure whether the correct thing is done but at least the reconstruction changed
		for(int i = 0; i < imgBufferSize; i = i + (2*gat_ign+1)){
			imgBuffer.getBuffer().put(0.0f);
			for(int j = 0; j < gat_ign; j++){
				imgBuffer.getBuffer().put(-100001.0f);
			}
			for(int j = 0; j < gat_ign; j++){
				imgBuffer.getBuffer().put(100001.0f);
			}
		}
		System.out.println(imgBuffer.getBuffer().toString());
		imgBuffer.getBuffer().rewind();
		//float[] bufferArray = imgBuffer.getBuffer().array(); doesnt work, dont know why because it was offered...
		
		//but the following method works
		/*for(int i = 0; i < imgBufferSize; i++){
			System.out.println("BufferArray at position " + i + "is: " + imgBuffer.getBuffer().get());
		}*/
		CLBuffer<FloatBuffer> projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		for(int p = 0; p < maxProjs; p++) {
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
		projMatrices.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue().putWriteBuffer(imgBuffer, false);
		queue.putWriteBuffer(projMatrices, true).finish();

		// copy params
		sino.show("pre for(p)");
		CLKernel kernel =  program.createCLKernel("backProjectPixelDrivenCL");
		for(int p = 0; p < maxProjs; p++) {
			
			//int gat_ign = this.gat_ign.clone();
		
			CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxU*maxV, Mem.READ_ONLY);
			for (int v=0;v<sino.getSize()[1];++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u <sino.getSize()[0]; u++) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v,p));
				}
			}
			sinoBuffer.getBuffer().rewind();
			CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
					sinoBuffer.getBuffer(), sino.getSize()[0], sino.getSize()[1],	//TODO MOEGLICHE FEHLERQULEL
					format);
			sinoBuffer.release();

			kernel.putArg(sinoGrid).putArg(imgBuffer).putArg(destBuffer).putArg(projMatrices)
				.putArg(p)
				.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
				.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
				.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ).putArg((int) gat_ign); 

			queue
				.putWriteImage(sinoGrid, true)
				.put2DRangeKernel(kernel, 0, 0, globalWorkSizeY, globalWorkSizeZ,
						localWorkSize, localWorkSize).putBarrier()
				.putReadBuffer(imgBuffer, true)
				.putReadBuffer(destBuffer, true)
				.finish();

			kernel.rewind();
			sinoGrid.release();
		}

		imgBuffer.getBuffer().rewind();
		destBuffer.getBuffer().rewind();
		/*for (int x=0; x < imgSizeX;++x) {	
			for (int y=0; y < imgSizeY;++y) {
				for(int z = 0; z< imgSizeZ; z++){
					//grid.setAtIndex(x, y, z, imgBuffer.getBuffer().get());
					grid.setAtIndex(x, y, z, destBuffer.getBuffer().get());
				}
			}
		}*/
		

		imgBuffer.getBuffer().rewind();
		float[] values = new float[imgSizeX * imgSizeY * imgSizeZ];
		//subtract the volumina of the 6 ignore volumina
		//create the correct grid afterwards
		int j = 0;
		float max = -Float.MAX_VALUE;
		float min = -Float.MIN_VALUE;
		for(int i = 0; i < imgBufferSize; i = i + (2*gat_ign+1)){
			float reconValue = imgBuffer.getBuffer().get(i);
			if(reconValue != 0.0){
				//System.out.println("ReconValue at  position " + i  + " is: " + reconValue);
			}
			for(int ign = 1; ign < 2*gat_ign + 1; ign++){
				float val = imgBuffer.getBuffer().get(i + ign);
				if(val < 100000.0f && val > -100000.0f){
					//subtract the values that shall be ignored
					if(val != 0.0f){
					//System.out.println("ReconValue: " + reconValue);
					//System.out.println("Value to subtract at position " + i + " is: " + val);
					reconValue = reconValue - val;
					}
				}
			}
			if(j < values.length){
				values[j] = reconValue;
				j++;
			}
		}
		
		j = 0;
		for (int x=0;x < imgSizeX;++x) {	
			for (int y=0;y < imgSizeY;++y) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int z = 0; z< imgSizeZ; z++){
					//grid.setAtIndex(x,y,z,values[j]);
					//if((values[j] <  0.0f  && values[j]|> | values[j] > 0.0f){
					if(values[j] > -0.000001f && values[j] < 0.000001f && values[j] != 0.0){
						//System.out.println("Values at position : " + j  + " is " + values[j]);
						//System.out.println("Values as double precision: " + (int) values[j]);
						NumberFormat formatter = new DecimalFormat("###.#########");  
						String f = formatter.format(values[j]);  
						//System.out.println("formatted: " + f);
						//System.out.println(String.format("%.16f", values[j]));
						//values[j] = 0.0f; 
					}
					grid.setAtIndex(x,y,z,values[j]);
					j ++;
				}
			}
		}

		// clean up
		imgBuffer.release();
		destBuffer.release();
		projMatrices.release();
		queue.release();
		kernel.release();
		program.release();
		context.release();

		grid.setSpacing(spacingX, spacingY, spacingZ);
		
		if (debug)
			System.out.println("Backprojection done.");
		return grid;
	}

	
	public int getGat_ign() {
		return gat_ign;
	}

	public void setGat_ign(int gat_ign) {
		this.gat_ign = gat_ign;
	}

	public Grid3D backprojectPixelDrivenCL(Grid2D sino, int projIdx) {
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
			System.err.println("ConeBeamBackprojector: Invalid projection index");
			return null;
		}
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		double spacingX = geo.getVoxelSpacingX();
		double spacingY = geo.getVoxelSpacingY();
		double spacingZ = geo.getVoxelSpacingZ();
		double originX = -geo.getOriginX();
		double originY = -geo.getOriginY();
		double originZ = -geo.getOriginZ();

		boolean debug = true;
		if (debug)
			System.out.println("Backprojecting...");
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
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeZ = OpenCLUtil.roundUp(localWorkSize, imgSizeZ); // rounded up to the nearest multiple of localWorkSize

		// load sources, create and build program
		CLProgram program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("ConeBeamBackProjector.cl"))
					.build();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(-1);
		}

		// create image from input grid
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		/*
		CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxProjs*maxU*maxV, Mem.READ_ONLY);
		//		for (int i = 0; i < grid.getSize()[0]; ++i) {
		//			imageBuffer.getBuffer().put(grid.getSubGrid(i).getBuffer());
		//		}

		for (int p=0;p<maxProjs;++p){
			for (int v=0;v<maxV;++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u < maxU; u++) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(p,v,u));
				}
			}
		}
		sinoBuffer.getBuffer().rewind();
		CLImage3d<FloatBuffer> sinoGrid = context.createImage3d(
				sinoBuffer.getBuffer(), maxU, maxV, maxProjs,	//TODO MOEGLICHE FEHLERQULEL
				format);
		sinoBuffer.release();
		*/

		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer((2*gat_ign+1)*imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		int imgBufferSize = (2*gat_ign+1)*imgSizeX*imgSizeY*imgSizeZ;
		
		//fillBuffer(imgBuffer.getBuffer(), gat_ign);didnt work properly
		for(int i = 0; i < imgBufferSize; i = i + (2*gat_ign+1)){
			imgBuffer.getBuffer().put(0.0f);
			for(int j = 0; j < gat_ign; j++){
				imgBuffer.getBuffer().put(-10001.0f);
			}
			for(int j = 0; j < gat_ign; j++){
				imgBuffer.getBuffer().put(10001.0f);
			}
		}
		System.out.println(imgBuffer.getBuffer().toString());

		imgBuffer.getBuffer().rewind();

		//create memory for projections
		CLBuffer<FloatBuffer> projMatrices = context.createFloatBuffer(1*3*4, Mem.READ_ONLY);//
		//for(int p = 0; p < maxProjs; p++) {
		int p = projIdx;
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		//}
		projMatrices.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue();
		queue.putWriteBuffer(projMatrices, true).finish();

		// copy params
		CLKernel kernel =  program.createCLKernel("backProjectPixelDrivenCL");
		//for(int p = 0; p < maxProjs; p++) {
			// create sinogram texture
			CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxU*maxV, Mem.READ_ONLY);
			for (int v=0;v<sino.getHeight();++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u < sino.getWidth(); u++) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v));//
				}
			}
			sinoBuffer.getBuffer().rewind();
			CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
					sinoBuffer.getBuffer(), maxU, maxV,	//TODO MOEGLICHE FEHLERQULEL
					format);
			sinoBuffer.release();

			kernel.putArg(sinoGrid).putArg(imgBuffer).putArg(projMatrices)
				.putArg(0)
				.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
				.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
				.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ); 

			queue
				.putWriteImage(sinoGrid, true)
				.put2DRangeKernel(kernel, 0, 0, globalWorkSizeY, globalWorkSizeZ,
						localWorkSize, localWorkSize).putBarrier()
				.putReadBuffer(imgBuffer, true)
				.finish();

			kernel.rewind();
			sinoGrid.release();
		//}

		/*imgBuffer.getBuffer().rewind();
		for (int x=0;x < imgSizeX;++x) {	
			for (int y=0;y < imgSizeY;++y) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int z = 0; z< imgSizeZ; z++){
					grid.setAtIndex(x,y,z,imgBuffer.getBuffer().get());
				}
			}
		}*/
		imgBuffer.getBuffer().rewind();
		
		
		float[] values = new float[imgSizeX * imgSizeY * imgSizeZ];
		//subtract the volumina of the 6 ignore volumina
		//create the correct grid afterwards
		int j = 0;
		for(int i = 0; i < imgBufferSize; i = i + (2*gat_ign+1)){
			float reconValue = imgBuffer.getBuffer().get();
			for(int ign = 0; ign < 2*gat_ign; ign++){
				float val = imgBuffer.getBuffer().get();
				if(val < 100000.0f || val > -100000.0f){
					System.out.println(val);
					//subtract the values that shall be ignored
					reconValue = reconValue - val;
				}
			}
			if(j < values.length){
				values[j] = reconValue;
				j++;
			}
		}
		
		j = 0;
		for (int x=0;x < imgSizeX;++x) {	
			for (int y=0;y < imgSizeY;++y) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int z = 0; z< imgSizeZ; z++){
					grid.setAtIndex(x,y,z,values[j]);
					if(values[j] < 0.0f && values[j] > 0.0f){
						System.out.println("Values at position : " + j  + " is " + values[j]);
					}
					j ++;
				}
			}
		}

		// clean up
		imgBuffer.release();
		projMatrices.release();
		queue.release();
		kernel.release();
		program.release();
		context.release();
		
		
		

		if (debug)
			System.out.println("Backprojection done.");
		return grid;
	}
	
	

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
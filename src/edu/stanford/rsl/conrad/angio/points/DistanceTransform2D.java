/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.points;

import ij.IJ;
import ij.ImageJ;

import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.BranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;
import edu.stanford.rsl.conrad.angio.util.image.ImageOps;

public class DistanceTransform2D {
	protected CLProgram program;
	protected CLKernel kernelFunction;
	protected CLCommandQueue commandQueue;	
	protected CLContext context;
	protected CLDevice device;

	protected float [] voxelSize = null;
	protected float [] volumeSize = null;
	protected float [] volumeOrigin = null;
		
	protected CLBuffer<FloatBuffer> gVolumeOrigin = null;
	protected CLBuffer<FloatBuffer> gVolumeSize = null;
	protected CLBuffer<FloatBuffer> gVoxelElementSize = null;
	protected CLBuffer<FloatBuffer> gPoints = null;
	
	static int bpBlockSize[] = {32, 32};
	int[] realLocalSize;
	int[] globalWorkSize;
	
	private OpenCLGrid2D slice2D = null;
	private ArrayList<?> points = null;
	
	private boolean configured = false;
	private boolean initialized = false;
	
	private boolean multiplySpacing = false;
	
	private boolean verbose = false;
	
	public static void main(String[] args){
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(".../hyst.tif"));
		Grid3D dT = new Grid3D(img);
		for(int k = 0; k < img.getSize()[2]; k++){
			ArrayList<PointND> pts = ImageOps.thresholdedPointList(img.getSubGrid(k), 0.5);
			DistanceTransform2D distTrafo = new DistanceTransform2D(img.getSubGrid(k), pts, true);
			Grid2D dist = distTrafo.run();
			dT.setSubGrid(k, dist);
		}
		new ImageJ();
		dT.show();
	}
	
	
	public DistanceTransform2D(Grid2D g, ArrayList<?> pts){
		this.slice2D = new OpenCLGrid2D(g);
		this.points = pts;
	}
	
	public DistanceTransform2D(Grid2D g, ArrayList<?> pts, boolean multSpac){
		this.slice2D = new OpenCLGrid2D(g);
		this.points = pts;
		this.multiplySpacing = multSpac;
	}

	public Grid2D run(){
		if(!configured){
			configure();
		}
		if(!initialized){
			init();
		}
		if(verbose)
			System.out.println("Calculating distance map.");
		// add kernel function to the queue
		commandQueue.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], 
							realLocalSize[0], realLocalSize[1]).finish();
		slice2D.getDelegate().notifyDeviceChange();
		if(verbose)
			System.out.println("Done.");
		return new Grid2D(slice2D);
	}
	
	private void configure(){
		if(verbose)
			System.out.println("Configuring.");
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		
		voxelSize = new float [2];
		volumeSize = new float [2];
		volumeOrigin = new float[2];
		
		voxelSize[0] = (float) slice2D.getSpacing()[0];
		voxelSize[1] = (float) slice2D.getSpacing()[1];
		volumeSize[0] = slice2D.getSize()[0];
		volumeSize[1] = slice2D.getSize()[1];
		volumeOrigin[0] = (float) slice2D.getOrigin()[0];
		volumeOrigin[1] = (float) slice2D.getOrigin()[1];
		
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		configured = true;
	}
	
	public void unload(){		
		slice2D.release();
		gPoints.release();
		gVoxelElementSize.release();
		gVolumeSize.release();
		gVolumeOrigin.release();
		kernelFunction.release();
		program.release();
		commandQueue.release();
//		context.release();
	}
	
	protected void init(){
		if(verbose)
			System.out.println("Initializing.");
		if (!initialized) {
			try {
				// get the fastest device
				commandQueue = device.createCommandQueue();

				// initialize the program
				if (program==null || !program.getContext().equals(context)){
					program = context.createProgram(this.getClass().getResourceAsStream("distanceTransform2DCL.cl")).build();
				}

				// (1) check space on device - At the moment we simply use 90% of the overall available memory
				long availableMemory =  (long)(device.getGlobalMemSize()*0.9);
				long requiredMemory = (long) Math.ceil((volumeSize[0] * volumeSize[1] * Float.SIZE/8)*2);
				if(verbose){
					System.out.println("Total available Memory on graphics card in MB: " + availableMemory/1024/1024);
					System.out.println("Required Memory on graphics card in MB: " + requiredMemory/1024/1024);
				}
				if (requiredMemory > availableMemory){
					throw new OutOfMemoryError("Not enough space on GPU, sorry");
				}

				// create the computing kernel
				this.createKernelFunction();

				gVolumeOrigin = context.createFloatBuffer(volumeOrigin.length, Mem.READ_ONLY);
				gVolumeOrigin.getBuffer().put(volumeOrigin);
				gVolumeOrigin.getBuffer().rewind();
								
				gVolumeSize = context.createFloatBuffer(volumeSize.length, Mem.READ_ONLY);
				gVolumeSize.getBuffer().put(volumeSize);
				gVolumeSize.getBuffer().rewind();
				
				gVoxelElementSize = context.createFloatBuffer(voxelSize.length, Mem.READ_ONLY);
				gVoxelElementSize.getBuffer().put(voxelSize);
				gVoxelElementSize.getBuffer().rewind();

				gPoints = context.createFloatBuffer(points.size()*2, Mem.READ_ONLY);
				gPoints.getBuffer().put(toFloatArray(points));
				gPoints.getBuffer().rewind();
				
				slice2D.getDelegate().prepareForDeviceOperation();
				
				commandQueue
				.putWriteBuffer(slice2D.getDelegate().getCLBuffer(), true) // writes the input texture image
				.putWriteBuffer(gVolumeOrigin, true)
				.putWriteBuffer(gVolumeSize,true)
				.putWriteBuffer(gVoxelElementSize,true)
				.putWriteBuffer(gPoints, true)
				.finish();

				this.writeKernelArguments();

				// make sure that product of 3D kernel side lengths is smaller than maxWorkGroupSize
				int maxWorkGroupSize = device.getMaxWorkGroupSize();
				realLocalSize = new int[]{	Math.min((int)Math.pow(maxWorkGroupSize,1/2.0), bpBlockSize[0]), 
						Math.min((int)Math.pow(maxWorkGroupSize,1/2.0), bpBlockSize[1])};

				// rounded up to the nearest multiple of localWorkSize
				globalWorkSize = new int[]{OpenCLUtil.roundUp(realLocalSize[0], (int)volumeSize[0]), 
						OpenCLUtil.roundUp(realLocalSize[1], (int)volumeSize[1])};			

			} catch (Exception e) {
				e.printStackTrace();
			}
			initialized = true;
		}
	}
	
	protected void writeKernelArguments() {
		// write kernel parameters
		kernelFunction.rewind();
		kernelFunction
		.putArg(slice2D.getDelegate().getCLBuffer())
		.putArg(gVolumeOrigin)
		.putArg(gVolumeSize)
		.putArg(gVoxelElementSize)
		.putArg(gPoints)
		.putArg(points.size());
	}

	
	private float[] toFloatArray(ArrayList<?> list){
		float[] arr = new float[list.size()*2];
		if(list.get(0) instanceof PointND && !multiplySpacing){		
			for(int i = 0; i < list.size(); i++){
				PointND p = (PointND) list.get(i);
				for(int j = 0; j < 2; j++){
					arr[i*2+j] = (float)p.get(j);
				}
			}
		}else if(list.get(0) instanceof PointND && multiplySpacing){		
			for(int i = 0; i < list.size(); i++){
				PointND p = (PointND) list.get(i);
				for(int j = 0; j < 2; j++){
					arr[i*2+j] = (float)p.get(j) * voxelSize[j];
				}
			}
		}else if(list.get(0) instanceof BranchPoint){
			for(int i = 0; i < list.size(); i++){
				BranchPoint p = (BranchPoint) list.get(i);
				arr[i*2+0] = (float)(p.x * voxelSize[0]);
				arr[i*2+1] = (float)(p.y * voxelSize[1]);
			}
		}else if(list.get(0) instanceof Point){
			for(int i = 0; i < list.size(); i++){
				Point p = (Point) list.get(i);
				arr[i*2+0] = (float)(p.x * voxelSize[0]);
				arr[i*2+1] = (float)(p.y * voxelSize[1]);
			}
		}else{
			System.err.println("Unknown point class. All points will be equal to zero.");
		}
		return arr;
	}
	
	
	protected void createKernelFunction() {
		kernelFunction = program.createCLKernel("distanceTransformKernel");
	}


	public boolean isVerbose() {
		return verbose;
	}


	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
}

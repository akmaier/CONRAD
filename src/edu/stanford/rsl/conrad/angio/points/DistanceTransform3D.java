/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.points;

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

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.angio.util.io.PointAndRadiusIO;

public class DistanceTransform3D {
	
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
	
	private OpenCLGrid3D volume3D = null;
	private ArrayList<PointND> points = null;
	
	private boolean configured = false;
	private boolean initialized = false;
	
	
	public static void main(String[] args){
		String pointsFile = ".../phase_90.reco3D";
		PointAndRadiusIO prio = new PointAndRadiusIO();
		prio.read(pointsFile);
		ArrayList<PointND> pts = prio.getPoints();
		
		int[] gSize = new int[]{256,256,300};
		double[] gSpace = new double[]{0.5,0.5,0.5};
		Grid3D g = new Grid3D(gSize[0], gSize[1], gSize[2]);
		g.setSpacing(gSpace);
		g.setOrigin(-(gSize[0]-1)*gSpace[0]/2,-(gSize[1]-1)*gSpace[1]/2,-(gSize[2]-1)*gSpace[2]/2);
		
		DistanceTransform3D cntl = new DistanceTransform3D(g, pts);
		Grid3D distMap = cntl.run();
		cntl.unload();
		
		new ImageJ();
		distMap.show();
	}
	
	
	public DistanceTransform3D(Grid3D g, ArrayList<PointND> pts){
		this.volume3D = new OpenCLGrid3D(g);
		this.points = pts;
	}

	public Grid3D run(){
		if(!configured){
			configure();
		}
		if(!initialized){
			init();
		}
		System.out.println("Calculating distance map.");
		// add kernel function to the queue
		commandQueue.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], 
							realLocalSize[0], realLocalSize[1]).finish();
		volume3D.getDelegate().notifyDeviceChange();
		System.out.println("Done.");
		return new Grid3D(volume3D);
	}
	
	private void configure(){
		System.out.println("Configuring.");
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		
		voxelSize = new float [3];
		volumeSize = new float [3];
		volumeOrigin = new float[3];
		
		voxelSize[0] = (float) volume3D.getSpacing()[0];
		voxelSize[1] = (float) volume3D.getSpacing()[1];
		voxelSize[2] = (float) volume3D.getSpacing()[2];
		volumeSize[0] = volume3D.getSize()[0];
		volumeSize[1] = volume3D.getSize()[1];
		volumeSize[2] = volume3D.getSize()[2];
		volumeOrigin[0] = (float) volume3D.getOrigin()[0];
		volumeOrigin[1] = (float) volume3D.getOrigin()[1];
		volumeOrigin[2] = (float) volume3D.getOrigin()[2];
		
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		configured = true;
	}
	
	public void unload(){		
		volume3D.release();
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
		System.out.println("Initializing.");
		if (!initialized) {
			try {
				// get the fastest device
				commandQueue = device.createCommandQueue();

				// initialize the program
				if (program==null || !program.getContext().equals(context)){
					program = context.createProgram(this.getClass().getResourceAsStream("distanceTransform3DCL.cl")).build();
				}

				// (1) check space on device - At the moment we simply use 90% of the overall available memory
				long availableMemory =  (long)(device.getGlobalMemSize()*0.9);
				long requiredMemory = (long) Math.ceil((volumeSize[0] * volumeSize[1] * volumeSize[2] * Float.SIZE/8)*2);
				System.out.println("Total available Memory on graphics card in MB: " + availableMemory/1024/1024);
				System.out.println("Required Memory on graphics card in MB: " + requiredMemory/1024/1024);
				if (requiredMemory > availableMemory){
					throw new OutOfMemoryError("Not enough space on GPU, sorry");
				}

				// create the computing kernel
				this.creatKernelFunction();

				gVolumeOrigin = context.createFloatBuffer(volumeOrigin.length, Mem.READ_ONLY);
				gVolumeOrigin.getBuffer().put(volumeOrigin);
				gVolumeOrigin.getBuffer().rewind();
								
				gVolumeSize = context.createFloatBuffer(volumeSize.length, Mem.READ_ONLY);
				gVolumeSize.getBuffer().put(volumeSize);
				gVolumeSize.getBuffer().rewind();
				
				gVoxelElementSize = context.createFloatBuffer(voxelSize.length, Mem.READ_ONLY);
				gVoxelElementSize.getBuffer().put(voxelSize);
				gVoxelElementSize.getBuffer().rewind();

				gPoints = context.createFloatBuffer(points.size()*3, Mem.READ_ONLY);
				gPoints.getBuffer().put(toFloatArray(points));
				gPoints.getBuffer().rewind();
				
				volume3D.getDelegate().prepareForDeviceOperation();
				
				commandQueue
				.putWriteBuffer(volume3D.getDelegate().getCLBuffer(), true) // writes the input texture image
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
		.putArg(volume3D.getDelegate().getCLBuffer())
		.putArg(gVolumeOrigin)
		.putArg(gVolumeSize)
		.putArg(gVoxelElementSize)
		.putArg(gPoints)
		.putArg(points.size());
	}

	
	private float[] toFloatArray(ArrayList<PointND> list){
		float[] arr = new float[list.size()*3];
		for(int i = 0; i < list.size(); i++){
			PointND p = list.get(i);
			for(int j = 0; j < 3; j++){
				arr[i*3+j] = (float)p.get(j);
			}
		}
		return arr;
	}
	
	protected void creatKernelFunction() {
		kernelFunction = program.createCLKernel("distanceTransformKernel");
	}
}

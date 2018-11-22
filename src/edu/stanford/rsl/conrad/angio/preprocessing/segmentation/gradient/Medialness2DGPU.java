/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.gradient;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class Medialness2DGPU {
	protected CLProgram program;
	protected CLKernel kernelFunction;
	protected CLCommandQueue commandQueue;	
	protected CLContext context;
	protected CLDevice device;

	protected float [] voxelSize = null;
	protected float [] volumeSize = null;
		
	protected CLBuffer<FloatBuffer> gVolumeSize = null;
	protected CLBuffer<FloatBuffer> gVoxelElementSize = null;
	
	protected CLBuffer<FloatBuffer> gSamplesPos = null;
	protected CLBuffer<FloatBuffer> gSamplesNeg = null;
	
	protected OpenCLGrid3D derivatives = null;
	private CLImage3d<FloatBuffer> gTexDerivatives = null;
	
	static int bpBlockSize[] = {32, 32};
	int[] realLocalSize;
	int[] globalWorkSize;
	
	private OpenCLGrid2D slice2D = null;
	
	private boolean configured = false;
	
	
	/** Scales are to be given in mm */
	private double[] scales = new double[]{0,0.5,1};
	
	ArrayList<float[]> filterBankDerivative = null;
	
	private Grid3D medialness = null;
		
	Grid3D stack = null;
	int[] gSize = null;
	double[] gSpace =  null;
	
	private double minR;
	private double maxR;
	private double dRadius = 0;
	private float[] lineSamplesPos = null; // in px
	private float[] lineSamplesNeg = null; // in px
	private int nRadii = 10;
	
	
	public static void main(String[] args){
		String testFile = ".../test.tif";
		
		new ImageJ();
		ImagePlus imp = IJ.openImage(testFile);
		ImageProcessor ip = imp.getProcessor();
		imp.setProcessor(ip);
		Grid3D stack = ImageUtil.wrapImagePlus(imp);
		//imp.show();
		
		Medialness2DGPU medialnessFilt = new Medialness2DGPU(stack, 1.0, 5, 10);
		medialnessFilt.setScales(new double[]{1.0,1.3,1.7});
		medialnessFilt.run();
		Grid3D med = medialnessFilt.getMedialnessImage();
		med.show();
		
	}

	public Medialness2DGPU(Grid3D stack, double minRadius, double maxRadius, int nRadii){
		
		this.stack = stack;		
		this.gSize = stack.getSize();
		double[] gSpacing = stack.getSpacing();
		if(gSpacing[0] != gSpacing[1]){
			//TODO RESAMPLE IMAGE TO SAME SPACING
			// Else, line extraction becomes complicated and directions are not properly defined
			gSpace = new double[]{gSpacing[0], gSpacing[0], gSpacing[2]};
		}else{
			gSpace = gSpacing;
		}
		this.minR = minRadius;
		this.maxR = maxRadius;
		this.nRadii = nRadii;
	}
	
		
	
	public void run(){
		
		Grid3D medRes = new Grid3D(gSize[0],gSize[1],gSize[2]);
		medRes.setSpacing(gSpace);
		
		init();
		configure();

		//for(int k = 0; k < 50; k++){	
		for(int k = 0; k < gSize[2]; k++){
			System.out.println("On slice "+String.valueOf(k+1)+" of "+String.valueOf(gSize[2])+".");
			initDerivatives(k);
			
			this.slice2D = new OpenCLGrid2D(new Grid2D(gSize[0], gSize[1]));
			slice2D.setSpacing(gSpace[0], gSpace[1]);
			
			Grid2D resp = runMedialnessCurrentSlice();
			FloatProcessor fp = ImageUtil.wrapGrid2D(resp);
			ShortProcessor sp = (ShortProcessor) fp.convertToShort(true);
			fp = (FloatProcessor) sp.convertToFloat();
			fp.multiply(1.0/65535.0);
			
			medRes.setSubGrid(k, ImageUtil.wrapFloatProcessor(fp));
		}
		
		this.medialness = medRes;
	}
	
	
	private void init(){
		setupFilterBank();
		
		this.dRadius = (maxR-minR) / nRadii;
		this.lineSamplesPos = new float[nRadii];
		this.lineSamplesNeg = new float[nRadii];
		
		for(int i = 0; i < nRadii; i++){
			lineSamplesPos[i] = (float)(+(minR + i*dRadius)/gSpace[0]);
			lineSamplesNeg[i] = (float)(-(minR + i*dRadius)/gSpace[0]);
		}
	}
	
	private void initDerivatives(int k){
		Grid2D slice = (Grid2D) this.stack.getSubGrid(k).clone();
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
		
		ImageStack derivStack = new ImageStack(gSize[0], gSize[1],scales.length*2);
		
		for(int i = 0; i < scales.length; i++){
			final int count = i;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {						
						Convolver c = null;				
						float[] derivative = filterBankDerivative.get(count);
						
						ImageProcessor gridAsImp = ImageUtil.wrapGrid2D(slice);
							
						ImageProcessor ip_x = gridAsImp.duplicate();
						c = new Convolver();
						c.convolveFloat(ip_x, derivative, derivative.length, 1);
						
						ImageProcessor ip_y = gridAsImp.duplicate();
						c = new Convolver();
						c.convolveFloat(ip_y, derivative, 1, derivative.length);			
						
						derivStack.setProcessor(ip_x, count*2+1);
						derivStack.setProcessor(ip_y, count*2+2);
					}
				}) // exe.submit()
			); // futures.add()							
		}
		for (Future<?> future : futures){
			   try{
			       future.get();
			   }catch (InterruptedException e){
			       throw new RuntimeException(e);
			   }catch (ExecutionException e){
			       throw new RuntimeException(e);
			   }
		}
		ImagePlus derivPlus = new ImagePlus();
		derivPlus.setStack(derivStack);
		this.derivatives = new OpenCLGrid3D(ImageUtil.wrapImagePlus(derivPlus));
	}
	
	private void setupFilterBank() {
		this.filterBankDerivative = new ArrayList<float[]>();
		for(int i = 0; i < scales.length; i++){
			double scale = scales[i];
			float[] derivative;
			if(scale == 0){
				derivative = new float[]{1,0,-1};
			}else{
				int filtSize = 1+2*4*(int)Math.ceil(scale/gSpace[0]);
				derivative = new float[filtSize];
				double denom = scale*scale;
				double norm = (1 / Math.sqrt(2*Math.PI*denom));
				for(int j = 0; j < filtSize; j++){
					double x = (j-(filtSize-1)/2)*gSpace[0];
					derivative[j] = (float)( (- x / denom) * norm * Math.exp(- 0.5 * x*x / denom) );
				}
			}
			filterBankDerivative.add(derivative);
		}		
	}

	
	public void setScales(double[] scales) {
		this.scales = scales;
	}

	private Grid2D runMedialnessCurrentSlice(){
		try{
			initCL();

			// add kernel function to the queue
			commandQueue
			.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], 
					realLocalSize[0], realLocalSize[1]).finish();

		} catch (Exception e) {
			// TODO: handle exception
			unload();
			e.printStackTrace();
		}

		slice2D.getDelegate().notifyDeviceChange();
		return new Grid2D(slice2D);
	}
	
	private void configure(){
		System.out.println("Configuring.");
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		
		voxelSize = new float [2];
		volumeSize = new float [2];
		
		voxelSize[0] = (float) gSpace[0];
		voxelSize[1] = (float) gSpace[1];
		volumeSize[0] = gSize[0];
		volumeSize[1] = gSize[1];
		
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		setConfigured(true);
	}
	
	protected void initCL(){
		System.out.println("Initializing.");
		try {
			// get the fastest device
			commandQueue = device.createCommandQueue();

			// initialize the program
			if (program==null || !program.getContext().equals(context)){
				program = context.createProgram(this.getClass().getResourceAsStream("medialness2DCL.cl")).build();
			}

			// (1) check space on device - At the moment we simply use 90% of the overall available memory
			long availableMemory =  (long)(device.getGlobalMemSize()*0.9);
			long requiredMemory = (long) Math.ceil((volumeSize[0] * volumeSize[1] * scales.length*2 * Float.SIZE/8)*2);
			System.out.println("Total available Memory on graphics card in MB: " + availableMemory/1024/1024);
			System.out.println("Required Memory on graphics card in MB: " + requiredMemory/1024/1024);
			if (requiredMemory > availableMemory){
				throw new OutOfMemoryError("Not enough space on GPU, sorry");
			}

			// create the computing kernel
			this.createKernelFunction();

			mapDerivativesTo3DOpenCLImage();
			
			gVolumeSize = context.createFloatBuffer(volumeSize.length, Mem.READ_ONLY);
			gVolumeSize.getBuffer().put(volumeSize);
			gVolumeSize.getBuffer().rewind();
			
			gVoxelElementSize = context.createFloatBuffer(voxelSize.length, Mem.READ_ONLY);
			gVoxelElementSize.getBuffer().put(voxelSize);
			gVoxelElementSize.getBuffer().rewind();

			gSamplesPos = context.createFloatBuffer(lineSamplesPos.length, Mem.READ_ONLY);
			gSamplesPos.getBuffer().put(lineSamplesPos);
			gSamplesPos.getBuffer().rewind();
			
			gSamplesNeg = context.createFloatBuffer(lineSamplesNeg.length, Mem.READ_ONLY);
			gSamplesNeg.getBuffer().put(lineSamplesNeg);
			gSamplesNeg.getBuffer().rewind();
					
			slice2D.getDelegate().prepareForDeviceOperation();
			
			commandQueue
			.putWriteBuffer(slice2D.getDelegate().getCLBuffer(), true) // writes the input texture image
			.putWriteBuffer(gVolumeSize,true)
			.putWriteBuffer(gVoxelElementSize,true)
			.putWriteBuffer(gSamplesPos, true)
			.putWriteBuffer(gSamplesNeg, true)
			.putWriteImage(gTexDerivatives, true)
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
	}
	
		
	/**
	 * release all CL related objects and free memory
	 */
	public void unload(){
		
		//release all buffers
		if (gTexDerivatives != null)
			gTexDerivatives.release();
		if (gVolumeSize != null)
			gVolumeSize.release();
		if (gSamplesNeg != null)
			gSamplesNeg.release();
		if (gSamplesPos != null)
			gSamplesPos.release();
		if (gVoxelElementSize != null)
			gVoxelElementSize.release();
		if (kernelFunction != null)
			kernelFunction.release();
		if (program != null)
			program.release();
		if (commandQueue != null && !commandQueue.isReleased())
			commandQueue.release();
	}
	
	protected void mapDerivativesTo3DOpenCLImage(){
		// make sure OpenCL is turned on / and things are on the device
		derivatives.getDelegate().prepareForDeviceOperation();

		// create the 3D volume texture
		// Create the 3D array that will contain the projection data
		// and will be accessed via the 3D texture
		derivatives.getDelegate().getCLBuffer().getBuffer().rewind();

		// set the texture
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		gTexDerivatives = context.createImage3d(derivatives.getDelegate().getCLBuffer().getBuffer(),
				(int)gSize[0], (int)gSize[1], (int)(2*scales.length), format, Mem.READ_ONLY);
		derivatives.getDelegate().release();
	}
	
	protected void createKernelFunction() {
		kernelFunction = program.createCLKernel("medialnessKernel");
	}
	
	protected void writeKernelArguments() {
		// write kernel parameters
		kernelFunction.rewind();
		kernelFunction
		.putArg(slice2D.getDelegate().getCLBuffer())
		.putArg(gVolumeSize)
		.putArg(gVoxelElementSize)
		.putArg(gSamplesPos)
		.putArg(gSamplesNeg)
		.putArg(lineSamplesPos.length)
		.putArg(gTexDerivatives)
		.putArg(scales.length);
	}
	
	public Grid3D getMedialnessImage(){
		return medialness;
	}

	public boolean isConfigured() {
		return configured;
	}

	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
	
}

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
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class ConeBeamBackprojector {

	final boolean debug = false;
	final boolean verbose = false;

	private static Trajectory geometry;
	
	public ConeBeamBackprojector() {
		Configuration.loadConfiguration();
		geometry = Configuration.getGlobalConfiguration().getGeometry();
	}
	
	public void configure(){
		geometry = Configuration.getGlobalConfiguration().getGeometry();
	}

	public void configure(Trajectory geom){
		geometry = geom;
	}
	
	public Grid3D backprojectPixelDriven(Grid2D sino, int projIdx) {
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		Projection[] projMats = geometry.getProjectionMatrices();
		int maxProjs = geometry.getProjectionStackSize();
		if(projIdx >= maxProjs || 0 > projIdx){
			System.err.println("ConeBeamBackprojector: Invalid projection index");
			return null;
		}
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
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
		Trajectory geo = geometry;
		final int imgSizeX = geo.getReconDimensionX();
		final int imgSizeY = geo.getReconDimensionY();
		final int imgSizeZ = geo.getReconDimensionZ();
		final Projection[] projMats = geo.getProjectionMatrices();
		final int maxProjs = geo.getProjectionStackSize();
		final Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		final double spacingX = geo.getVoxelSpacingX();
		final double spacingY = geo.getVoxelSpacingY();
		final double spacingZ = geo.getVoxelSpacingZ();
		final double originX = -geo.getOriginX();
		final double originY = -geo.getOriginY();
		final double originZ = -geo.getOriginZ();
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
		
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		Projection[] projMats = geometry.getProjectionMatrices();
		int maxProjs = geometry.getProjectionStackSize();
		
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		
		if (debug)
			System.out.println("Backprojecting...");
		// create context
		CLContext context = OpenCLUtil.getStaticContext();
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

		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize

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

		/* optimization regarding number of function calls
		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		imgBuffer.getBuffer().rewind();
		CLBuffer<FloatBuffer> projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		final FloatBuffer projMatricesBuffer = projMatrices.getBuffer();
		for(int p = 0; p < maxProjs; p++) {
			final SimpleMatrix currentProjMatrix = projMats[p].computeP();
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatricesBuffer.put((float)currentProjMatrix.getElement(row, col));
					// one line version:
					//projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
		*/
		
		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = volume.getDelegate().getCLBuffer();//context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		
		CLBuffer<FloatBuffer> projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		for(int p = 0; p < maxProjs; p++) {
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
				
		projMatrices.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue();//.putWriteBuffer(imgBuffer, false);
		queue.putWriteBuffer(projMatrices, true).finish();

		// copy params
		CLKernel kernel =  program.createCLKernel("backProjectPixelDrivenCL");
		for(int p = 0; p < maxProjs; p++) {
			
			CLBuffer<FloatBuffer> sinoBuffer = sino[p].getDelegate().getCLBuffer();
			/*for (int v=0;v<sino.getSize()[1];++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u <sino.getSize()[0]; u++) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v,p));
				}
			}
			sinoBuffer.getBuffer().rewind();*/
			//TODO MOEGLICHE FEHLERQUELLE
			CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(sinoBuffer.getBuffer(), sino[p].getSize()[0], sino[p].getSize()[1],format,Mem.READ_ONLY);
			//sinoBuffer.release();

			kernel.putArg(sinoGrid)
			    .putArg(imgBuffer)
			    .putArg(projMatrices)
				.putArg(p)
				.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
				.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
				.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ); 

			queue
				.putWriteImage(sinoGrid, true)
				.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,
						localWorkSize, localWorkSize).putBarrier()
				//.putReadBuffer(imgBuffer, true)
				.finish();

			kernel.rewind();
			//sinoGrid.release();
		}

		float D = (float) geometry.getSourceToDetectorDistance();
		float scal = (float)(geometry.getSourceToAxisDistance() / geometry.getSourceToDetectorDistance());
		NumericPointwiseOperators.multiplyBy(volume, (float) (D * D	* Math.PI*scal / geometry.getNumProjectionMatrices()));
		
		/*imgBuffer.getBuffer().rewind();
		for (int x=0; x < imgSizeX;++x) {	
			for (int y=0; y < imgSizeY;++y) {
				for(int z = 0; z< imgSizeZ; z++){
					grid.setAtIndex(x, y, z, imgBuffer.getBuffer().get());
				}
			}
		}
		imgBuffer.getBuffer().rewind();


		// clean up
		imgBuffer.release();*/
		projMatrices.release();
		queue.release();
		kernel.release();
		program.release();
		//context.release();
		
		if (debug || verbose)
			System.out.println("Backprojection done.");

		
	}
	
	public Grid3D backprojectPixelDrivenCL(Grid3D sino) {
		
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		
		OpenCLGrid2D [] sinoCL = new OpenCLGrid2D[sino.getSize()[2]];
		for (int i=0; i < sinoCL.length; i++) 
			sinoCL[i] = new OpenCLGrid2D(sino.getSubGrid(i));
		
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		OpenCLGrid3D gridCL = new OpenCLGrid3D(grid);
		gridCL.getDelegate().prepareForDeviceOperation();
		
		backprojectPixelDrivenCL(gridCL, sinoCL);
		gridCL.setOrigin(-originX, -originY, -originZ);
		gridCL.setSpacing(spacingX, spacingY, spacingZ);
		for (int i=0; i < sinoCL.length; i++) sinoCL[i].release();
		grid = new Grid3D(gridCL);
		gridCL.release();
		return grid;
	}
	
	public Grid3D backprojectPixelDrivenCL(Grid2D sino , int projIdx) {
		
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		
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
		return grid;
	}
	
	public void backprojectPixelDrivenCL(OpenCLGrid3D volume, OpenCLGrid2D sino, int projIdx) {
		
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		Projection[] projMats = geometry.getProjectionMatrices();
		int maxProjs = geometry.getProjectionStackSize();
		
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		
		if (debug)
			System.out.println("Backprojecting...");
		// create context
		CLContext context = OpenCLUtil.getStaticContext();
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

		// Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		int globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); // rounded up to the nearest multiple of localWorkSize

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

		/* optimization regarding number of function calls
		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		imgBuffer.getBuffer().rewind();
		CLBuffer<FloatBuffer> projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		final FloatBuffer projMatricesBuffer = projMatrices.getBuffer();
		for(int p = 0; p < maxProjs; p++) {
			final SimpleMatrix currentProjMatrix = projMats[p].computeP();
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatricesBuffer.put((float)currentProjMatrix.getElement(row, col));
					// one line version:
					//projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
		*/
		
		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = volume.getDelegate().getCLBuffer();//context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		
		CLBuffer<FloatBuffer> projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		for(int p = 0; p < maxProjs; p++) {
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
				
		projMatrices.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue();//.putWriteBuffer(imgBuffer, false);
		queue.putWriteBuffer(projMatrices, true).finish();

		// copy params
		CLKernel kernel =  program.createCLKernel("backProjectPixelDrivenCL");
		
			
			CLBuffer<FloatBuffer> sinoBuffer = sino.getDelegate().getCLBuffer();
			/*for (int v=0;v<sino.getSize()[1];++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u <sino.getSize()[0]; u++) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v,p));
				}
			}
			sinoBuffer.getBuffer().rewind();*/
			//TODO MOEGLICHE FEHLERQUELLE
			CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(sinoBuffer.getBuffer(), sino.getSize()[0], sino.getSize()[1],format,Mem.READ_ONLY);
			//sinoBuffer.release();

			kernel.putArg(sinoGrid)
			    .putArg(imgBuffer)
			    .putArg(projMatrices)
				.putArg(projIdx)
				.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
				.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
				.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ); 

			queue
				.putWriteImage(sinoGrid, true)
				.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,
						localWorkSize, localWorkSize).putBarrier()
				//.putReadBuffer(imgBuffer, true)
				.finish();

			kernel.rewind();
			//sinoGrid.release();


		float D = (float) geometry.getSourceToDetectorDistance();
		float scal = (float)(geometry.getSourceToAxisDistance() / geometry.getSourceToDetectorDistance());
		NumericPointwiseOperators.multiplyBy(volume, (float) (D * D	* Math.PI*scal / geometry.getNumProjectionMatrices()));
	
		projMatrices.release();
		queue.release();
		kernel.release();
		program.release();
		
	}
	
	/*public Grid3D backprojectPixelDrivenCL(Grid2D sino, int projIdx) {
		geometry = Configuration.getGlobalConfiguration().getGeometry();
		int maxV = geometry.getDetectorHeight();
		int maxU = geometry.getDetectorWidth();
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		Projection[] projMats = geometry.getProjectionMatrices();
		int maxProjs = geometry.getProjectionStackSize();
		if(projIdx >= maxProjs || 0 > projIdx){
			System.err.println("ConeBeamBackprojector: Invalid projection index");
			return null;
		}
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		grid.setOrigin(-originX, -originY, -originZ);
		grid.setSpacing(spacingX, spacingY, spacingZ);
		
		if (debug)
			System.out.println("Backprojecting...");
		// create context
		CLContext context = OpenCLUtil.getStaticContext();
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

		// create memory for image
		CLBuffer<FloatBuffer> imgBuffer = context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.WRITE_ONLY);
		imgBuffer.getBuffer().rewind();
		CLBuffer<FloatBuffer> projMatrix = context.createFloatBuffer(1*3*4, Mem.READ_ONLY);//
		//for(int p = 0; p < maxProjs; p++) {
		int p = projIdx;
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrix.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		//}
		projMatrix.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue().putWriteBuffer(imgBuffer, false);
		queue.putWriteBuffer(projMatrix, true).finish();

		// copy params
		CLKernel kernel =  program.createCLKernel("backProjectPixelDrivenCL");

			CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxU*maxV, Mem.READ_ONLY);
			for (int v=0; v<sino.getSize()[1]; ++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u=0; u<sino.getSize()[0]; ++u) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v));//
				}
			}
			sinoBuffer.getBuffer().rewind();
			CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
					sinoBuffer.getBuffer(), sino.getSize()[0], sino.getSize()[1],	//TODO MOEGLICHE FEHLERQUELLE
					format, Mem.READ_ONLY);
			sinoBuffer.release();

			kernel.putArg(sinoGrid).putArg(imgBuffer).putArg(projMatrix)
				.putArg(0) // just one projection, therefore, take the first (and only) projection matrix
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


		imgBuffer.getBuffer().rewind();
		for (int x=0; x<imgSizeX; ++x) {	
			for (int y=0; y<imgSizeY; ++y) {
				for(int z=0; z<imgSizeZ; ++z){
					grid.setAtIndex(x,y,z,imgBuffer.getBuffer().get());
				}
			}
		}
		imgBuffer.getBuffer().rewind();

		// clean up
		imgBuffer.release();
		projMatrix.release();
		queue.release();
		kernel.release();
		program.release();
		context.release();

		if (debug || verbose)
			System.out.println("Backprojection done.");
		return grid;
	}
	*/
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
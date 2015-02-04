package edu.stanford.rsl.tutorial.iterative;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImage3d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.cone.ConeBeamProjector;

public class SartCL extends SartCPU{
	
	// controll structures
	CLContext context = null;
	CLDevice device = null;
	//CLCommandQueue queueFP = null;
	//CLCommandQueue queueBP = null;
	CLProgram programFP = null;
	CLProgram programBP = null;
	CLImageFormat format = null;
	
	CLKernel kernelFP = null;
	CLKernel kernelBP = null;
	
	// memory for image(s)
	CLBuffer<FloatBuffer> volCLFP = null;		// volume returned after SART iterations
	CLBuffer<FloatBuffer> volCLBP = null;	// intermediate volume used for updating. BP fills and resets this
	
	// memory for projections
	CLBuffer<FloatBuffer> projCLFP = null;
	CLBuffer<FloatBuffer> projCLBP = null;
	//CLBuffer<FloatBuffer> oProjCL = null;
	//CLBuffer<FloatBuffer> normProjCL = null;
	
	// memory for projection matrices
	CLBuffer<FloatBuffer> projMatrices = null;
	
	private int imgSizeX;
	private int imgSizeY;
	private int imgSizeZ;
	private Projection[] projMats = null;
	private int maxProjs;
	private int maxU;
	private float spacingU;
	private float spacingV;
	private int maxV;
	private float spacingX;
	private float spacingZ;
	private float spacingY;
	private float originX;
	private float originZ;
	private float originY;
	private int imageSize;
	private int localWorkSizeCLFP;
	private int globalWorkSizeUCLFP;
	private int globalWorkSizeVCLFP;
	private int localWorkSizeCLBP;
	private int globalWorkSizeYCLBP;
	private int globalWorkSizeZCLBP;
	
	/**
	 * This constructor takes the following arguments:
	 * @param volDims
	 * @param spacing
	 * @param origin
	 * @param oProj
	 * @param beta
	 * @throws Exception
	 */
	public SartCL(int[] volDims, double[] spacing, double[] origin,
			Grid3D oProj, float beta) throws Exception {
		super(volDims, spacing, origin, oProj, beta);
		initCLDatastructure();
	}
	
	public SartCL(Grid3D initialVol, Grid3D sino, float beta) throws Exception {
		super(initialVol, sino, beta);
		initCLDatastructure();
	}

	public final void iterate() throws Exception{
		iterate(1);
	}
	
	public final void iterate(final int iter) throws Exception {
		for (int i = 0; i < iter; ++i) {

			boolean[] projIsUsed = new boolean[maxProjs]; // default: false
			int p = 0; // current projection index

			for (int n = 0; n < maxProjs; ++n) {
				/* edit/init data structures */
				projIsUsed[p] = true;
				Grid2D sino = projectRayDrivenCL(vol, p);
				//sino = gop.transpose(sino);
				
				//sino.show("sino p=" + p + " i=" + i);
				if(debug && 0 < gop.normL1(vol)){
					//sino.show("Sinogram p=" + p + " i=" + i);
					ConeBeamProjector cbp = new ConeBeamProjector();
					Grid3D sinoTest = USE_CL_FP ? cbp.projectRayDrivenCL(vol) : cbp
							.projectPixelDriven(vol);
					Grid2D sinoTestP = sinoTest.getSubGrid(p);
					sinoTest.show("sinoCL-Test");
					sinoTestP.show("sinoCL-Test-Proj:" + p);
					Grid2D s = new Grid2D(sinoTestP);
					gop.subtractBySave(s, sino);
					s.show("sinoCL-Test-Proj-Diff");
					System.out.println("Diff L1: " + gop.normL1(s));
				}
				gop.fillInvalidValues(sino, 0);
				
				if (verbose) System.out.println("Projection range: "+ gop.min(sino) + ":" + gop.max(sino)); // TEST
				
				//oProj.show("oProj");
				
				Grid2D oProjP = new Grid2D(oProj.getSubGrid(p));
				Grid2D normSinoP = new Grid2D(normSino.getSubGrid(p)); // used read-only, cloning not necessary but save
				gop.multiplyBy(normSinoP, normFactor);

				/* update step */
				// NOTE: upd = (oProj - sino) ./ normSino
				
				if (verbose) reportInvalidValues(oProjP, "oProjP"); // Just in case.. 
				if (verbose) reportInvalidValues(sino, "sino"); // Just in case.. should not happen after fillInvalidValues()
				
				gop.subtractBy(oProjP, sino);
				
				if (verbose) reportInvalidValues(oProjP, "oP-si"); // Just in case.. 
				if (verbose) reportInvalidValues(normSinoP, "normP"); // Just in case.. 
				
				gop.divideBySave(oProjP, normSinoP);
				Grid2D upd = oProjP;

				if (verbose) reportInvalidValues(upd, "for proj " + p);

				// NOTE: vol = vol + updBP * beta
				// upd.setOrigin(oProj.getOrigin()[0], oProj.getOrigin()[1]); //
				// needed after update?
				// upd.setSpacing(oProj.getSpacing()[0], oProj.getSpacing()[1]);
				// // needed after update?
				
				//upd.show("Update projection");
				Grid3D updBP = backprojectPixelDrivenCL(upd, p);

				if (verbose) reportInvalidValues(updBP, "updBP");
				//updBP.show("updBP p=" + p);
				
				if(debug){
					ConeBeamBackprojector cbbp = new ConeBeamBackprojector();
					Grid3D updBPTest = USE_CL_BP ? cbbp.backprojectPixelDrivenCL(upd, p)
							: cbbp.backprojectPixelDriven(upd, p);
					if (verbose) reportInvalidValues(updBPTest, "updBPTest");
					gop.multiplyBySave(updBPTest, 1);
					updBPTest.show("updBPTest");
					
					System.out.println("vol: " + Arrays.toString(vol.getOrigin()));
					System.out.println("updBP: " + Arrays.toString(updBP.getOrigin()));
					System.out.println("updBPTest: " + Arrays.toString(updBPTest.getOrigin()));
				}				
				
				// GridOp.addInPlace(vol, GridOp.mulInPlace(updBP, beta));
				gop.multiplyBySave(updBP, beta);
				if (verbose) reportInvalidValues(updBP, "updBP after mult");
				
				gop.addBy(vol, updBP);
				if (verbose) reportInvalidValues(vol, "vol after " + i + " SART iterations");

				/*
				 * Don't use projections with a small angle to each other
				 * subsequently
				 */
				p = (p + maxProjs / 3) % maxProjs;
				for (int ii = 1; projIsUsed[p] && ii < maxProjs; ++ii)
					p = (p + 1) % maxProjs;
			}
		}
		releaseCLMemory();
	}
	
	private void initCLDatastructure(){
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory geo = conf.getGeometry();
		
		maxV = geo.getDetectorHeight();
		maxU = geo.getDetectorWidth();
		spacingU = (float) geo.getPixelDimensionX();
		spacingV = (float) geo.getPixelDimensionY();
		imgSizeX = geo.getReconDimensionX();
		imgSizeY = geo.getReconDimensionY();
		imgSizeZ = geo.getReconDimensionZ();
		projMats = conf.getGeometry().getProjectionMatrices();
		maxProjs = conf.getGeometry().getProjectionStackSize();
		spacingX = (float) geo.getVoxelSpacingX();
		spacingY = (float) geo.getVoxelSpacingY();
		spacingZ = (float) geo.getVoxelSpacingZ();
		originX = (float) -geo.getOriginX();
		originY = (float) -geo.getOriginY();
		originZ = (float) -geo.getOriginZ();
		
		// create context
		context = OpenCLUtil.createContext();
		// select device
		device = context.getMaxFlopsDevice();
		
		// create programs
		try {
			programFP = context.createProgram(this.getClass().getResourceAsStream("../cone/ConeBeamProjector.cl"))
					.build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		try {
			programBP = context.createProgram(this.getClass().getResourceAsStream("../cone/ConeBeamBackProjector.cl"))
					.build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);

		imageSize = imgSizeX*imgSizeY*imgSizeZ;
		// Length of arrays to process
		localWorkSizeCLFP = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		globalWorkSizeUCLFP = OpenCLUtil.roundUp(localWorkSizeCLFP, maxU); // rounded up to the nearest multiple of localWorkSize
		globalWorkSizeVCLFP = OpenCLUtil.roundUp(localWorkSizeCLFP, maxV); // rounded up to the nearest multiple of localWorkSize

		// Length of arrays to process
		localWorkSizeCLBP = Math.min(device.getMaxWorkGroupSize(), 8); // Local work size dimensions
		globalWorkSizeYCLBP = OpenCLUtil.roundUp(localWorkSizeCLBP, imgSizeY); // rounded up to the nearest multiple of localWorkSize
		globalWorkSizeZCLBP = OpenCLUtil.roundUp(localWorkSizeCLBP, imgSizeZ); // rounded up to the nearest multiple of localWorkSize
		
		// create memory for image
		//volCL = context.createFloatBuffer(imgSizeX*imgSizeY*imgSizeZ, Mem.READ_WRITE);
		volCLFP = context.createFloatBuffer(imageSize, Mem.READ_WRITE);
		volCLBP = context.createFloatBuffer(imageSize, Mem.WRITE_ONLY);
		
		projMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		projMatrices.getBuffer().rewind();
		for(int p = 0; p < maxProjs; p++) {
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					projMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
		projMatrices.getBuffer().rewind();
		
		// create memory for sinogram
		projCLBP = context.createFloatBuffer(maxU * maxV , Mem.READ_ONLY);
		projCLFP = context.createFloatBuffer(maxU * maxV , Mem.WRITE_ONLY);
		
		
		//CLBuffer<FloatBuffer> volCL = null;
		//CLBuffer<FloatBuffer> updVolCL = null; // TODO necessary?
		//	OK	CLImage3d<FloatBuffer> volCLImageGrid = null;
		
		// memory for projections
		//	OK	CLBuffer<FloatBuffer> projCL = null;
		//CLBuffer<FloatBuffer> oProjCL = null;
		//CLBuffer<FloatBuffer> normProjCL = null;
		
		// memory for projection matrices
		//	OK	CLBuffer<FloatBuffer> projMatrix = null;

	}

	private void releaseCLMemory(){
		// clean up
		projCLFP.release();
		projCLBP.release();
		
		volCLFP.release();
		volCLBP.release();
		
		projMatrices.release();
		
		if(null != kernelFP) kernelFP.release();
		if(null != kernelBP) kernelBP.release();
		programFP.release();
		programBP.release();
		context.release();
	}
	
	public Grid2D projectRayDrivenCL(Grid3D grid, int projIdx) {
		copyGridToCLVol(vol, volCLFP);
		CLImage3d<FloatBuffer> volCLImageGrid = createCLImage3dFromCLBuffer(volCLFP); // OLD: imageGrid
		//volCL.release();
		CLCommandQueue queueFP = device.createCommandQueue();
		queueFP.putWriteImage(volCLImageGrid, true).finish(); // TODO update imageGrid first
		// copy params
		if(null == kernelFP)
			kernelFP =  programFP.createCLKernel("projectRayDrivenCL");
		
		Grid2D sino = new Grid2D(maxU,maxV); // sino will be returned by method // TODO U and V right?
		
		//for(int p = 0; p < maxProjs; p++) {
		int p = projIdx;
			
			SimpleVector source = projMats[p].computeCameraCenter();
			SimpleVector pAxis = projMats[p].computePrincipalAxis();
			kernelFP.putArg(volCLImageGrid).putArg(projCLFP)//.putArg(projMatrices)
			.putArg(p)
			.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
			.putArg(originX).putArg(originY).putArg(originZ)
			.putArg(spacingX).putArg(spacingY).putArg(spacingZ)
			.putArg(maxU).putArg(maxV)
			.putArg(spacingU).putArg(spacingV)
			.putArg((float)source.getElement(0)).putArg((float)source.getElement(1)).putArg((float)source.getElement(2))
			.putArg((float)pAxis.getElement(0)).putArg((float)pAxis.getElement(1)).putArg((float)pAxis.getElement(2)); 

			queueFP
			.put2DRangeKernel(kernelFP, 0, 0, globalWorkSizeUCLFP, globalWorkSizeVCLFP,
					localWorkSizeCLFP, localWorkSizeCLFP).putBarrier()
					.putReadBuffer(projCLFP, true)
					.finish();
			
			copyCLSinoToGrid(projCLFP, sino); // TODO evtl. v,u statt u,v
			/*
			projCLFP.getBuffer().rewind();
			for (int v=0;v < maxV;++v) {			//TODO MOEGLICHE FEHLERQUELLE
				for(int u = 0; u< maxU; u++){
					sino.setAtIndex(v,u, projCLFP.getBuffer().get());
				}
			}
			projCLFP.getBuffer().rewind();
			*/
			kernelFP.rewind();
		//}
		if (debug || verbose)
			System.out.println("Projection done!");
	
		volCLImageGrid.release();
		queueFP.release();
		return sino;
	}
	
	
	public Grid3D backprojectPixelDrivenCL(Grid2D sino, int projIdx) {
		if(projIdx >= maxProjs || 0 > projIdx){
			System.err.println("ConeBeamBackprojector: Invalid projection index");
			return null;
		}
		
		if (debug)
			System.out.println("Backprojecting...");
		
		// create/reuse memory for image
		fillCLGridWith(volCLBP, 0);
		
		projMatrices.getBuffer().rewind();
		// write buffers to the device
		CLCommandQueue queueBP = device.createCommandQueue().putWriteBuffer(volCLBP, false);
		queueBP.putWriteBuffer(projMatrices, true).finish();

		// copy params
		if(null == kernelBP)
			kernelBP =  programBP.createCLKernel("backProjectPixelDrivenCL");
		int p = projIdx;
		//for(int p = 0; p < maxProjs; p++) {
			// create sinogram texture
			copyGridToCLSino(sino, projCLBP);
			/*
			CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxU*maxV, Mem.READ_ONLY);
			for (int v=0; v<sino.getSize()[1]; ++v) {
				for(int u=0; u<sino.getSize()[0]; ++u) {
					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v));//
				}
			}
			sinoBuffer.getBuffer().rewind();
			*/
			CLImage2d<FloatBuffer> sinoGrid = createCLImage2dFromCLBuffer(projCLBP);
			//sinoBuffer.release(); // projCLBP.release();

			kernelBP.putArg(sinoGrid).putArg(volCLBP).putArg(projMatrices)
				.putArg(p) // just one projection, therefore, take the first (and only) projection matrix
				.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
				.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
				.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ); 

			queueBP
				.putWriteImage(sinoGrid, true)
				.put2DRangeKernel(kernelBP, 0, 0, globalWorkSizeYCLBP, globalWorkSizeZCLBP,
						localWorkSizeCLBP, localWorkSizeCLBP).putBarrier()
				.putReadBuffer(volCLBP, true)
				.finish();

			kernelBP.rewind();
			sinoGrid.release();
		//}

		// prepare grid to return
		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ); // grid will be returned by method
		grid.setOrigin(-originX, -originY, -originZ);
		grid.setSpacing(spacingX, spacingY, spacingZ);
		copyCLVolToGrid(volCLBP, grid);
		
		queueBP.release();

		if (debug || verbose)
			System.out.println("Backprojection done.");
		return grid;
	}
	
	private void copyCLVolToGrid(CLBuffer<FloatBuffer> imgBuffer, Grid3D grid){
		imgBuffer.getBuffer().rewind();
		for (int x=0; x<imgSizeX; ++x)
			for (int y=0; y<imgSizeY; ++y)
				for(int z=0; z<imgSizeZ; ++z)
					grid.setAtIndex(x,y,z,imgBuffer.getBuffer().get());
		imgBuffer.getBuffer().rewind();
	}
	
	private void copyGridToCLVol(Grid3D grid, CLBuffer<FloatBuffer> imgBuffer){
		imgBuffer.getBuffer().rewind();
		for(int z=0; z<imgSizeZ; ++z)
			for (int y=0; y<imgSizeY; ++y)
				for (int x=0; x<imgSizeX; ++x)
					imgBuffer.getBuffer().put(grid.getAtIndex(x,y,z));
		imgBuffer.getBuffer().rewind();
	}
	
	private void copyGridToCLSino(Grid2D sino, CLBuffer<FloatBuffer> sinoBuffer){
		sinoBuffer.getBuffer().rewind();
		for (int v=0; v<sino.getSize()[1]; ++v) {
			for(int u=0; u<sino.getSize()[0]; ++u) {
				sinoBuffer.getBuffer().put(sino.getAtIndex(u,v));
			}
		}
		sinoBuffer.getBuffer().rewind();
	}
	
	private void copyCLSinoToGrid(CLBuffer<FloatBuffer> sinoBuffer, Grid2D sino){
		sinoBuffer.getBuffer().rewind();
		for (int v=0; v<sino.getSize()[1]; ++v)
			for(int u=0; u<sino.getSize()[0]; ++u)
				sino.setAtIndex(u,v,sinoBuffer.getBuffer().get()); // TODO right order?
		sinoBuffer.getBuffer().rewind();
	}
	
	private CLImage3d<FloatBuffer> createCLImage3dFromCLBuffer(
			CLBuffer<FloatBuffer> imageBuffer) {
		CLImage3d<FloatBuffer> imageGrid = context.createImage3d(
				imageBuffer.getBuffer(), imgSizeX, imgSizeY, imgSizeZ,
				format);
		imageGrid.getBuffer().rewind();
		return imageGrid;
	}
	
	private CLImage2d<FloatBuffer> createCLImage2dFromCLBuffer(CLBuffer<FloatBuffer> sinoBuffer){
		CLImage2d<FloatBuffer> sinoGrid = context.createImage2d(
				sinoBuffer.getBuffer(), maxU, maxV, format); // TODO u and v ok?
		// sino.getSize()[0], sino.getSize()[1]
		sinoGrid.getBuffer().rewind();
		return sinoGrid;
	}
	
	private void fillCLGridWith(CLBuffer<FloatBuffer> imgBuffer, float val) {
		imgBuffer.getBuffer().rewind();
		for(int i=0; i<imageSize; ++i)
			imgBuffer.getBuffer().put(val);
		imgBuffer.getBuffer().rewind();
	}

//	public Grid2D projectRayDrivenCL(Grid3D grid) {
//		copyGridToCLVol(vol, volCLFP);
//		CLImage3d<FloatBuffer> volCLImageGrid = createCLImage3dFromCLBuffer(volCLFP); // OLD: imageGrid
//		//volCL.release();
//		CLCommandQueue queueFP = device.createCommandQueue();
//		queueFP.putWriteImage(volCLImageGrid, true).finish(); // TODO update imageGrid first
//		// copy params
//		if(null == kernelFP)
//			kernelFP =  programFP.createCLKernel("projectRayDrivenCL");
//		
//		Grid3D sino = new Grid3D(maxU,maxV,maxProjs); // sino will be returned by method // TODO U and V right?
//		
//		for(int p = 0; p < maxProjs; p++) {
//			
//			SimpleVector source = projMats[p].computeCameraCenter();
//			SimpleVector pAxis = projMats[p].computePrincipalAxis();
//			kernelFP.putArg(volCLImageGrid).putArg(projCLFP)//.putArg(projMatrices)
//			.putArg(p)
//			.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
//			.putArg(originX).putArg(originY).putArg(originZ)
//			.putArg(spacingX).putArg(spacingY).putArg(spacingZ)
//			.putArg(maxU).putArg(maxV)
//			.putArg(spacingU).putArg(spacingV)
//			.putArg((float)source.getElement(0)).putArg((float)source.getElement(1)).putArg((float)source.getElement(2))
//			.putArg((float)pAxis.getElement(0)).putArg((float)pAxis.getElement(1)).putArg((float)pAxis.getElement(2)); 
//
//			queueFP
//			.put2DRangeKernel(kernelFP, 0, 0, globalWorkSizeUCLFP, globalWorkSizeVCLFP,
//					localWorkSizeCLFP, localWorkSizeCLFP).putBarrier()
//					.putReadBuffer(projCLFP, true)
//					.finish();
//			
//			copyCLSinosToGrid(projCLFP, sino); // TODO evtl. v,u statt u,v
//			/*
//			projCLFP.getBuffer().rewind();
//			for (int v=0;v < maxV;++v) {			//TODO MOEGLICHE FEHLERQUELLE
//				for(int u = 0; u< maxU; u++){
//					sino.setAtIndex(v,u, projCLFP.getBuffer().get());
//				}
//			}
//			projCLFP.getBuffer().rewind();
//			*/
//			kernelFP.rewind();
//		}
//		if (debug || verbose)
//			System.out.println("Projection done!");
//	
//		volCLImageGrid.release();
//		queueFP.release();
//		return sino;
//	}
//	
//	
//	public Grid3D backprojectPixelDrivenCL(Grid2D sino, int projIdx) {
//		if(projIdx >= maxProjs || 0 > projIdx){
//			System.err.println("ConeBeamBackprojector: Invalid projection index");
//			return null;
//		}
//		
//		if (debug)
//			System.out.println("Backprojecting...");
//		
//		// create/reuse memory for image
//		fillCLGridWith(volCLBP, 0);
//		
//		projMatrices.getBuffer().rewind();
//		// write buffers to the device
//		CLCommandQueue queueBP = device.createCommandQueue().putWriteBuffer(volCLBP, false);
//		queueBP.putWriteBuffer(projMatrices, true).finish();
//
//		// copy params
//		if(null == kernelBP)
//			kernelBP =  programBP.createCLKernel("backProjectPixelDrivenCL");
//		int p = projIdx;
//		//for(int p = 0; p < maxProjs; p++) {
//			// create sinogram texture
//			copyGridToCLSino(sino, projCLBP);
//			/*
//			CLBuffer<FloatBuffer> sinoBuffer = context.createFloatBuffer(maxU*maxV, Mem.READ_ONLY);
//			for (int v=0; v<sino.getSize()[1]; ++v) {
//				for(int u=0; u<sino.getSize()[0]; ++u) {
//					sinoBuffer.getBuffer().put(sino.getAtIndex(u,v));//
//				}
//			}
//			sinoBuffer.getBuffer().rewind();
//			*/
//			CLImage2d<FloatBuffer> sinoGrid = createCLImage2dFromCLBuffer(projCLBP);
//			//sinoBuffer.release(); // projCLBP.release();
//
//			kernelBP.putArg(sinoGrid).putArg(volCLBP).putArg(projMatrices)
//				.putArg(p) // just one projection, therefore, take the first (and only) projection matrix
//				.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
//				.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
//				.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ); 
//
//			queueBP
//				.putWriteImage(sinoGrid, true)
//				.put2DRangeKernel(kernelBP, 0, 0, globalWorkSizeYCLBP, globalWorkSizeZCLBP,
//						localWorkSizeCLBP, localWorkSizeCLBP).putBarrier()
//				.putReadBuffer(volCLBP, true)
//				.finish();
//
//			kernelBP.rewind();
//			sinoGrid.release();
//		//}
//
//		// prepare grid to return
//		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ); // grid will be returned by method
//		grid.setOrigin(-originX, -originY, -originZ);
//		grid.setSpacing(spacingX, spacingY, spacingZ);
//		copyCLVolToGrid(volCLBP, grid);
//		
//		queueBP.release();
//
//		if (debug || verbose)
//			System.out.println("Backprojection done.");
//		return grid;
//	}
	
}

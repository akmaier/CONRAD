package edu.stanford.rsl.conrad.angio.motion.respiratory;

import ij.IJ;
import ij.ImageJ;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;

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

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.PreprocessingPipeline;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;

public class MaxErrorBackProjCL {

	//image variables
	private int imgSizeX;
	private int imgSizeY;
	private int imgSizeZ;
	private Projection[] projMats;
	private int maxProjs;
	private float spacingX;
	private float spacingY;
	private float spacingZ;
	private float originX;
	private float originY;
	private float originZ;
	private ArrayList<float[]> shifts = null;
	private OpenCLGrid3D projs;
	
	//cl variables
	private CLContext context;
	private CLDevice device;
	private CLBuffer<FloatBuffer> gProjMatrices;
	private CLBuffer<FloatBuffer> gShiftsX;
	private CLBuffer<FloatBuffer> gShiftsY;
	private CLBuffer<FloatBuffer> gShiftsZ;
	private CLCommandQueue queue;
	private CLKernel kernel;
	// Length of arrays to process
	private int localWorkSize;
	private int globalWorkSizeX; 
	private int globalWorkSizeY; 
	private CLImageFormat format;
	CLImage3d<FloatBuffer> sinoGrid;
	private CLProgram program;
	
	
	public static void main(String[] args) {
		int caseID = 3;		
		double refHeartPhase = 0.8;
		
		int[] volSize = new int[]{256, 256, 256};
		float[] volSpace = new float[]{0.5f, 0.5f, 0.5f};
				
		DataSets datasets = DataSets.getInstance();
		DataSet ds = datasets.getCase(caseID);
		
		String outputDir = ds.getDir()+"eval/";
		
		Angiogram prepAng;
		String dir = outputDir + String.valueOf(refHeartPhase)+"/";
		File fTest = new File(dir);
		if(!fTest.exists()){
			PreprocessingPipeline prepPipe = new PreprocessingPipeline(ds);
			prepPipe.setRefHeartPhase(refHeartPhase);
			prepPipe.evaluate();
			prepAng = prepPipe.getPreprocessedAngiogram();
		}else{
			Projection[] ps = ProjMatIO.readProjMats(dir+"pMat.txt");
			Grid3D cm = ImageUtil.wrapImagePlus(IJ.openImage(dir+"distTrafo.tif"));
			Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(dir+"img.tif"));
			prepAng = new Angiogram(img, ps, new double[ps.length]);
			prepAng.setReconCostMap(cm);
			ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
			Grid3D binImg = SkeletonUtil.costMapToVesselTreeImage(cm);
			for(int i = 0; i < ps.length; i++){
				skels.add(SkeletonUtil.binaryImgToSkel(binImg.getSubGrid(i), 0, false));
			}
			prepAng.setSkeletons(skels);
		}
		
		ArrayList<float[]> shifts = new ArrayList<float[]>();
		for(int i = 0; i < prepAng.getPMatrices().length; i++){
			shifts.add(new float[3]);
		}
		//shifts.set(3,new float[]{0,0,1});
		
		MaxErrorBackProjCL maxErrBP = new MaxErrorBackProjCL(volSize, volSpace, 
				prepAng.getPMatrices(), prepAng.getReconCostMap(), shifts);
		Grid3D errorMap = maxErrBP.backprojectCL();
		maxErrBP.unload();
		
		new ImageJ();
		errorMap.show();
	
	}
	
	public MaxErrorBackProjCL(int[] gSz, float[] gSp, Projection[] pMats, Grid3D projs) {
		configure(gSz,gSp,pMats, projs);
		initCL();
	}
	
	public MaxErrorBackProjCL(int[] gSz, float[] gSp, Projection[] pMats, Grid3D projs, ArrayList<float[]> shift) {
		this.shifts = shift;
		configure(gSz,gSp,pMats, projs);
		initCL();
	}

	public void configure(int[] gSz, float[] gSp, Projection[] pMats, Grid3D projections){
		imgSizeX = gSz[0];
		imgSizeY = gSz[1];
		imgSizeZ = gSz[2];
		projMats = pMats;
		maxProjs = pMats.length;
		spacingX = gSp[0];
		spacingY = gSp[1];
		spacingZ = gSp[2];
		originX = -(gSz[0]-1)/2f*gSp[0];
		originY = -(gSz[1]-1)/2f*gSp[1];
		originZ = -(gSz[2]-1)/2f*gSp[2];
		projs = new OpenCLGrid3D(projections);
	}
	
	private void initCL(){
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();

		// load sources, create and build program
		program = null;
		try {
			program = context.createProgram(this.getClass().getResourceAsStream("MaxErrorBP.cl")).build();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		kernel =  program.createCLKernel("backProjectPixelDrivenCL");
		
		// create image from input grid
		format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		projs.getDelegate().getCLBuffer().getBuffer().rewind();
		sinoGrid = context.createImage3d(projs.getDelegate().getCLBuffer().getBuffer(),
				projs.getSize()[0], projs.getSize()[1], projs.getSize()[2], format, Mem.READ_ONLY);
		projs.getDelegate().release();
		queue.putWriteImage(sinoGrid, true).finish();
		
		localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16);
		globalWorkSizeX = OpenCLUtil.roundUp(localWorkSize, imgSizeX); 
		globalWorkSizeY = OpenCLUtil.roundUp(localWorkSize, imgSizeY); 
		
		gProjMatrices = context.createFloatBuffer(maxProjs*3*4, Mem.READ_ONLY);
		for(int p = 0; p < maxProjs; p++) {
			for(int row = 0; row < 3; row++) {
				for(int col = 0; col < 4; col++) {
					gProjMatrices.getBuffer().put((float)projMats[p].computeP().getElement(row, col));
				}
			}
		}
		gProjMatrices.getBuffer().rewind();
		queue.putWriteBuffer(gProjMatrices, true).finish();
		
		gShiftsX = context.createFloatBuffer(maxProjs, Mem.READ_ONLY);
		gShiftsY = context.createFloatBuffer(maxProjs, Mem.READ_ONLY);
		gShiftsZ = context.createFloatBuffer(maxProjs, Mem.READ_ONLY);
		if(shifts != null){
			for(int p = 0; p < maxProjs; p++) {
				gShiftsX.getBuffer().put(shifts.get(p)[0]);
				gShiftsY.getBuffer().put(shifts.get(p)[1]);
				gShiftsZ.getBuffer().put(shifts.get(p)[2]);
			}
		}else{
			for(int p = 0; p < maxProjs; p++) {
				gShiftsX.getBuffer().put(0f);
				gShiftsY.getBuffer().put(0f);
				gShiftsZ.getBuffer().put(0f);
			}
		}		
		gShiftsX.getBuffer().rewind();
		gShiftsY.getBuffer().rewind();
		gShiftsZ.getBuffer().rewind();
		queue.putWriteBuffer(gShiftsX, true).finish();
		queue.putWriteBuffer(gShiftsY, true).finish();
		queue.putWriteBuffer(gShiftsZ, true).finish();
		
	}
	
	public void unload(){
		if(gShiftsZ != null && !gShiftsZ.isReleased())
			gShiftsZ.release();
		if(gShiftsY != null && !gShiftsY.isReleased())
			gShiftsY.release();
		if(gShiftsX != null && !gShiftsX.isReleased())
			gShiftsX.release();
		if(gProjMatrices != null && !gProjMatrices.isReleased())
			gProjMatrices.release();
		if(sinoGrid != null && !sinoGrid.isReleased())
			sinoGrid.release();
		if(kernel != null && !kernel.isReleased())
			kernel.release();
		if(program != null && !program.isReleased())
			program.release();
		if(queue != null && !queue.isReleased())
			queue.release();		
	}

	public Grid3D backprojectCL() {

		Grid3D grid = new Grid3D(imgSizeX,imgSizeY,imgSizeZ);
		grid.setSpacing(spacingX,spacingY,spacingZ);
		grid.setOrigin(originX,originY,originZ);
		//NumericPointwiseOperators.fill(grid, 1);
		OpenCLGrid3D gridCL = new OpenCLGrid3D(grid);
		gridCL.getDelegate().prepareForDeviceOperation();
			
		kernel.putArg(sinoGrid)
		.putArg(gridCL.getDelegate().getCLBuffer())
		.putArg(gProjMatrices)
		.putArg(gShiftsX)
		.putArg(gShiftsY)
		.putArg(gShiftsZ)
		.putArg(maxProjs)
		.putArg(imgSizeX).putArg(imgSizeY).putArg(imgSizeZ)
		.putArg((float)originX).putArg((float)originY).putArg((float)originZ)
		.putArg((float)spacingX).putArg((float)spacingY).putArg((float)spacingZ); 

		queue
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,localWorkSize, localWorkSize)
		.finish();
			
		gridCL.getDelegate().notifyDeviceChange();
		grid = new Grid3D(gridCL);
		gridCL.release();
		return grid;
	}


	public ArrayList<float[]> getShifts() {
		return shifts;
	}

	public void setShifts(ArrayList<float[]> shifts) {
		this.shifts = shifts;
	}

}
	


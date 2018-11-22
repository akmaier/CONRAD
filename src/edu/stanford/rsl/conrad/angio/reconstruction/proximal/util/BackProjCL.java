package edu.stanford.rsl.conrad.angio.reconstruction.proximal.util;

import java.io.IOException;
import java.nio.FloatBuffer;
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

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class BackProjCL {

	final boolean debug = false;
	final boolean verbose = false;

	private boolean configured = false;
	
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

	public BackProjCL(int[] gSize, double[] gSpace, double[] gOrigin, Projection[] pMat) {
		configure(gSize, gSpace, gOrigin, pMat);
		initCL();
	}

	public void configure(int[] gSize, double[] gSpace, double[] gOrigin, Projection[] pMat){
		imgSizeX = gSize[0];
		imgSizeY = gSize[1];
		imgSizeZ = gSize[2];
		projMats = pMat;
		maxProjs = pMat.length;
		spacingX = gSpace[0];
		spacingY = gSpace[1];
		spacingZ = gSpace[2];
		originX  = -gOrigin[0];
		originY  = -gOrigin[1];
		originZ  = -gOrigin[2];
		configured = true;
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
			.putArg(1f); 

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
		if(configured){
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
		return null;
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
		.putArg(1f); 

		queue
		.putCopyBufferToImage(sino.getDelegate().getCLBuffer(), sinoGrid).finish()
		.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY,localWorkSize, localWorkSize)
		.finish();

		kernel.rewind();
		
		volume.getDelegate().notifyDeviceChange();
		sinoGrid.release();

	}

}
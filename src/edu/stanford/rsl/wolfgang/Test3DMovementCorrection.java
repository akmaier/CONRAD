package edu.stanford.rsl.wolfgang;

import java.io.IOException;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.io.FileInfo;
import ij.io.FileOpener;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class Test3DMovementCorrection {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		
//		int wA = 2;
//		int hA = 3;
//		int wB = 4;
//		int hB = 2;
//		
//		
//		
//		
//		ComplexGrid2D matrixA = new ComplexGrid2D(wA,hA);
//		ComplexGrid2D matrixB = new ComplexGrid2D(wB,hB);
//		matrixA.activateCL();
//		matrixB.activateCL();
//		
//		for(int i = 0;-9.0295021E8 i < wA; i++){
//			for(int j = 0; j < hA; j++){
//				matrixA.setImagAtIndex(1.0f, i, j);
//			}
//		}
//		
//		for(int i = 0; i < wB; i++){
//			for(int j = 0; j < hB; j++){
//				matrixB.setImagAtIndex(1.0f, i, j);
//			}
//		}
//		
//		
//		System.out.println("\n Matrix A");
//		for(int i = 0; i < matrixA.getSize()[1]; i++){
//			for(int j =0; j < matrixA.getSize()[0]; j++){
//				System.out.print(matrixA.getRealAtIndex(j, i) + " + " + matrixA.getImagAtIndex(j,i) + "i,  ");
//			}
//			System.out.println("");
//		}
//		
//		System.out.println("\n Matrix B");
//		for(int i = 0; i < matrixB.getSize()[1]; i++){
//			for(int j =0; j < matrixB.getSize()[0]; j++){
//				System.out.print(matrixB.getRealAtIndex(j, i) + " + " + matrixA.getImagAtIndex(j,i) + "i,  ");
//			}
//			System.out.println("");
//		}
//		
//		
//		
//		ComplexGrid2D result = matMulComplex(matrixA, matrixB);
//		
//		//System.out.println(result.getAtIndex(0, 0));
//
//		System.out.println("\n result matrix");
//		for(int i = 0; i < result.getSize()[1]; i++){
//			for(int j =0; j < result.getSize()[0]; j++){
//				System.out.print(result.getRealAtIndex(j, i) + " + " + result.getImagAtIndex(j,i) + "i,  ");
//			}
//			System.out.println("");
//		}
//		System.out.println("ready");
//		
		
		
		
		
		new ImageJ();
		
		// read in projection data
		Grid3D projections = null;
		try {
			// locate the file
			// here we only want to select files ending with ".bin". This will open them as "Dennerlein" format.
			// Any other ImageJ compatible file type is also OK.
			// new formats can be added to HandleExtraFileTypes.java
			String filenameString = FileUtil.myFileChoose("proj/ciptmp/co98jaha/workspace/data/FinalProjections80kev/FORBILD_Head_80kev.tif",".tif", false);
			// call the ImageJ routine to open the image:
			ImagePlus imp = IJ.openImage(filenameString);
			// Convert from ImageJ to Grid3D. Note that no data is copied here. The ImageJ container is only wrapped. Changes to the Grid will also affect the ImageJ ImagePlus.
			projections = ImageUtil.wrapImagePlus(imp);
			// Display the data that was read from the file.
			//projections.show("Data from file");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		NumericPointwiseOperators.fill(projections,0f);
		projections.setAtIndex(0, 0, 0, 1);
		//projections.show("Data from file");
		// get configuration
		String xmlFilename = "/proj/ciptmp/co98jaha/workspace/data/ConradSettingsForbild3D.xml";
		Config conf = new Config(xmlFilename,0, 2);
		conf.getMask().show("mask");
		System.out.println("N: "+ conf.getHorizontalDim() + " M: " + conf.getVerticalDim() + " K: "+  conf.getNumberOfProjections());
		Grid1D testShift = new Grid1D(conf.getNumberOfProjections()*2);
		//testShift.setAtIndex(0, 50.0f);
//		for(int i = 0; i < testShift.getSize()[0]; i++){
//			if(i%2 == 0){
//				testShift.setAtIndex(i, 1.0f);
//			}
//			else{
//				testShift.setAtIndex(i, 1.0f);
//			}
//		}

		
		MovementCorrection3D mc = new MovementCorrection3D(projections, conf, false);
		mc.setShiftVector(testShift);
		mc.doFFT2();
		//mc.getData().show("2D-Fouriertransformed before transposing");
		mc.transposeData();
		
		//ComplexGrid3D nextGrid = (ComplexGrid3D) mc.get2dFourierTransposedData().clone();


//		mc.doFFTAngle();
//		mc.get2dFourierTransposedData().show("fft before");
//		mc.doiFFTAngle();
//		
//		mc.doFFTAngleCL();
//		mc.get3dFourier().show("dft");
//		mc.doiFFTAngleCL();
//		
		

		//mc.doiFFTAngleCL();
		
		Grid1D result = mc.computeOptimalShift();
		for(int i = 0; i < result.getNumberOfElements(); i++){
			float val = result.getAtIndex(i);
			System.out.println(i + ": " + val);
		}
		result.show("Result shift vector");
		
//		for(int i = 0; i < 2; i++){
//			
//			mc.doFFTAngleCL();
//			//mc.get3dFourier().getRealGrid().show("Real Grid after forward " + i);
//			//mc.get3dFourier().getImagGrid().show("Imag Grid after forward " + i);
//			//mc.get3dFourier().getRealGrid().show();
//			ComplexGrid3D  fftCLGrid = mc.get3dFourier();
//	//		mc.doFFTAngle();
//	//		ComplexGrid3D fftGrid = mc.get2dFourierTransposedData();
//	//		mc.doiFFTAngle();
//			Grid2D mask = conf.getMask();
//			float sumCL = 0;
//			float sumCPU = 0;
//			int counter = 0;
//			
//			for(int proj = 0; proj < fftCLGrid.getSize()[0]; proj++){
//				for(int u = 0; u < fftCLGrid.getSize()[1]; u++){
//					if(mask.getAtIndex(proj, u) == 1){
//						counter++;
//						for(int v = 0; v < fftCLGrid.getSize()[2]; v++){
//							sumCL += fftCLGrid.getAtIndex(proj, u, v).getReal();
//	//						sumCPU += fftGrid.getAtIndex(proj, u, v).getReal();
//						} 
//					}
//				}
//			}
//			System.out.println("Anzahl erkannte 1 in Maske: " + counter);
//			System.out.println("Anzahl v: " + fftCLGrid.getSize()[2]);
//			int numElCl = fftCLGrid.getNumberOfElements();
//	//		int numEl = fftGrid.getNumberOfElements();
//			System.out.println("CLGrid size: " + numElCl);// + ", cpuGrid size:" + numEl);
//			//sumCL /= numElCl;
//			//sumCPU /= numEl;
//			System.out.println("normalized sum cl: "+ sumCL);// + ", sum cpu" + sumCPU);
//			//mc.doFFTAngle();
//			//mc.get2dFourierTransposedData().show("FFT");
//			//mc.get3dFourier().show("DFT");
//			mc.doiFFTAngleCL();
//		}
		
		//mc.get2dFourierTransposedData().show("after angle backtransform");
		
//		
////		//mc.computeOptimalShift();
		//mc.applyShift();
//		long time = System.currentTimeMillis();
//		for(int i = 0; i < 30; i++){
//			mc.parallelShiftOptimized();
//		}
//		time = System.currentTimeMillis() - time;
//		System.out.println("complete time for 30 shifts: " + time);
//		time /= 30;
//		System.out.println("average time per shift:" + time);
////		
		mc.backTransposeData();
//		//mc.getData().show("2D-Fouriertransformed after transposing");
		mc.doiFFT2();
		mc.getData().getRealGrid().show("Real Data");
		mc.getData().getImagGrid().show("Imag Data");
		mc.getData().show("After pipeline");
	
//		
		
		
		
		
//		//mc.getData().show();
////		mc.get2dFourierTransposedData().show("Before angle fft");
//		time = System.currentTimeMillis();
//		mc.doFFTAngle();
////			
//		mc.doiFFTAngle();
//		time = System.currentTimeMillis();
//		System.out.println("time for fft on angles and back: " + time);
//		mc.get2dFourierTransposedData().show("After angle ifft");
//		mc.get2dFourierTransposedData().show("3d fft");
		
//		
//		mc.doiFFT2();
//	
//		mc.getData().show("After pipeline");
//
		
		
//		mc.getData().getRealGrid().show("backtransformed, real part");
//		mc.getData().getImagGrid().show("backtransformed, imag part");
//		double[] sbt = mc.getData().getSpacing();
//		System.out.println(sbt[0]+", "+sbt[1]+", "+sbt[2]);
//		
		
//		Grid1D uSpacingVec = conf.getUSpacingVec();
//		Grid1D vSpacingVec = conf.getVSpacingVec();
//		Grid1D kSpacingVec = conf.getKSpacingVec();
//		
//		uSpacingVec.show();
//		vSpacingVec.show();
//		kSpacingVec.show();
		

//		
		

//		try {
//			String fileSaveString = FileUtil.myFileChoose("proj/ciptmp/co98jaha/workspace/data/FinalProjections80kev/FORBILD_Head_80kev_1DFourier","tif", true);
//			ImageUtil.saveAs(projNewOrder, fileSaveString);
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}


//		
		
//		System.out.println("N: "+ conf.getHorizontalDim() + " M: " + conf.getVerticalDim() + " K: "+  conf.getNumberOfProjections());
		

	}
	public static Grid2D matmul(Grid2D grid1, Grid2D grid2){
		Grid2D result = new Grid2D(grid2.getWidth(),grid1.getHeight());
		
		OpenCLGrid2D clGrid1= new OpenCLGrid2D(grid1);
		OpenCLGrid2D clGrid2 = new OpenCLGrid2D(grid2);
		OpenCLGrid2D clGridResult = new OpenCLGrid2D(result);

		
		CLBuffer<FloatBuffer> buffGrid1 = clGrid1.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> buffGrid2 = clGrid2.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> buffGridResult = clGridResult.getDelegate().getCLBuffer();
		
		clGrid1.getDelegate().prepareForDeviceOperation();
		clGrid2.getDelegate().prepareForDeviceOperation();
		clGridResult.getDelegate().prepareForDeviceOperation();
		
		// create Context
		CLContext context = OpenCLUtil.getStaticContext();
		// choose fastest device
		CLDevice device = context.getMaxFlopsDevice();
		CLProgram program = null;
		try {
		program = context.createProgram(Test3DMovementCorrection.class.getResourceAsStream("matrixMul.cl"));
		} catch (IOException e) {
		e.printStackTrace();
		}
		program.build();
		
		
		CLKernel kernel = program.createCLKernel("matrixMul");
		kernel.putArg(buffGridResult);
		kernel.putArg(buffGrid1);
		kernel.putArg(buffGrid2);
		kernel.putArg(grid1.getWidth());
		kernel.putArg(grid1.getHeight());
		kernel.putArg(grid2.getWidth());
		kernel.putArg(grid2.getHeight());	
		
		int localWorksize = 2;
		long globalWorksizeA = OpenCLUtil.roundUp(localWorksize, result.getWidth());
		long globalWorksizeB = OpenCLUtil.roundUp(localWorksize, result.getHeight());
		
		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put2DRangeKernel(kernel, 0, 0, globalWorksizeA, globalWorksizeB, localWorksize, localWorksize).finish();
		
		clGrid1.getDelegate().notifyDeviceChange();
		clGrid2.getDelegate().notifyDeviceChange();
		clGridResult.getDelegate().notifyDeviceChange();
		
		result = clGridResult;
		
		
		

		return result;
	}
	
	public static ComplexGrid2D matMulComplex(ComplexGrid2D grid1, ComplexGrid2D grid2){
		ComplexGrid2D result = new ComplexGrid2D(grid2.getSize()[0],grid1.getSize()[1]);
		result.activateCL();
		CLBuffer<FloatBuffer> bufferRes = result.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferA  = grid1.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferB = grid2.getDelegate().getCLBuffer();
		
		result.getDelegate().prepareForDeviceOperation();
		grid1.getDelegate().prepareForDeviceOperation();
		grid2.getDelegate().prepareForDeviceOperation();
		
		// create Context
		CLContext context = OpenCLUtil.getStaticContext();
		// choose fastest device
		CLDevice device = context.getMaxFlopsDevice();
		CLProgram program = null;
		try {
		program = context.createProgram(Test3DMovementCorrection.class.getResourceAsStream("matrixMul.cl"));
		} catch (IOException e) {
		e.printStackTrace();
		}
		program.build();
		
		CLKernel kernel = program.createCLKernel("complexMatrixMul");
		kernel.putArg(bufferRes);
		kernel.putArg(bufferA);
		kernel.putArg(bufferB);
		kernel.putArg(grid1.getSize()[0]);
		kernel.putArg(grid1.getSize()[1]);
		kernel.putArg(grid2.getSize()[0]);
		kernel.putArg(grid2.getSize()[1]);	
		
		int localWorksize = 2;
		long globalWorksizeA = OpenCLUtil.roundUp(localWorksize, result.getSize()[0]);
		long globalWorksizeB = OpenCLUtil.roundUp(localWorksize, result.getSize()[1]);
		
		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put2DRangeKernel(kernel, 0, 0, globalWorksizeA, globalWorksizeB, localWorksize, localWorksize).finish();
		
		grid1.getDelegate().notifyDeviceChange();
		grid2.getDelegate().notifyDeviceChange();
		result.getDelegate().notifyDeviceChange();		
		
		return result;
	}
	

	


}

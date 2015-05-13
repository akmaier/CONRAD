package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

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
		
		//NumericPointwiseOperators.fill(projections,0f);
		//projections.setAtIndex(0, 0, 0, 1);
		projections.show("Data from file");
		
		// get configuration
		String xmlFilename = "/proj/ciptmp/co98jaha/workspace/data/ConradSettingsForbild3D.xml";
		Config conf = new Config(xmlFilename,15, 1);
		conf.getMask().show("mask");
		System.out.println("N: "+ conf.getHorizontalDim() + " M: " + conf.getVerticalDim() + " K: "+  conf.getNumberOfProjections());
		Grid1D testShift = new Grid1D(conf.getNumberOfProjections()*2);

		// Test shifting of 1 pixel each projection, both dimensions
		
		// 
		for(int i = 0; i < testShift.getSize()[0]; i++){
			if(i%2 == 0){
				testShift.setAtIndex(i, 1.0f);
			}
			else{
				testShift.setAtIndex(i, 1.0f);
			}
		}

		
		MovementCorrection3D mc = new MovementCorrection3D(projections, conf,false);
		mc.setShiftVector(testShift);
		mc.doFFT2();
		//mc.getData().show("2D-Fouriertransformed before transposing");
		mc.transposeData();
//		for(int i = 0; i < 10000; i++){
//			mc.parallelShiftOptimized();
//			System.out.println("Runde:" + i);
//		}
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
	}
	
}

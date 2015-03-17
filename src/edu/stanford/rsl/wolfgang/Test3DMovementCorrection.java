package edu.stanford.rsl.wolfgang;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.io.FileInfo;
import ij.io.FileOpener;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
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
			projections.show("Data from file");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		// get configuration
		String xmlFilename = "/proj/ciptmp/co98jaha/workspace/data/ConradSettingsForbild3D.xml";
		Config conf = new Config(xmlFilename, 7);
		conf.getMask().show("mask");
		Grid1D testShift = new Grid1D(conf.getNumberOfProjections()*2);
		//testShift.setAtIndex(0, 1.0f);
//		for(int i = 0; i < testShift.getSize()[0]; i++){
//			if(i%2 == 0){
//				testShift.setAtIndex(i, 1.0f);
//			}
//			else{
//				testShift.setAtIndex(i, 1.0f);
//			}
//		}
		
//		
		MovementCorrection3D mc = new MovementCorrection3D(projections, conf);
		mc.setShiftVector(testShift);
		mc.doFFT2();
		//mc.getData().show("2D-Fouriertransformed");
		mc.transposeData();
		
		mc.computeOptimalShift();
		//mc.applyShift();
//		long time = System.currentTimeMillis();
//		for(int i = 0; i < 50; i++){
//			mc.parallelizedApplyShift2D();
//		}
//		time = System.currentTimeMillis() - time;
//		System.out.println("complete time for 50 shifts: " + time);
//		time /= 50;
//		System.out.println("average time per shift:" + time);
//		//mc.getData().show();
////		mc.get2dFourierTransposedData().show("Before angle fft");
//		time = System.currentTimeMillis();
//		mc.doFFTAngle();
////			
//		mc.doiFFTAngle();
//		time = System.currentTimeMillis();
//		System.out.println("time for fft on angles and back: " + time);
//		mc.get2dFourierTransposedData().show("After angle ifft");
		mc.backTransposeData();
		
//		
		mc.doiFFT2();
//		
		mc.getData().show("After pipeline");
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
		
		System.out.println("N: "+ conf.getHorizontalDim() + " M: " + conf.getVerticalDim() + " K: "+  conf.getNumberOfProjections());
		

	}
	
	public static Grid3D changeDimOrder(Grid3D orig){
		
		int[] sizeOrig = orig.getSize();
		Grid3D gridChangedDim = new Grid3D(sizeOrig[2],sizeOrig[0], sizeOrig[1]);
		gridChangedDim.setSpacing(orig.getSpacing()[2], orig.getSpacing()[0],orig.getSpacing()[1]);
		gridChangedDim.setOrigin(orig.getOrigin());
		for(int angle = 0; angle < sizeOrig[2]; angle++){
			for(int horiz = 0; horiz < sizeOrig[0]; horiz++){
				for(int vert = 0; vert < sizeOrig[1]; vert++){
					//float value = orig.getAtIndex(horiz, vert, angle);
					//if(value > 0){
						gridChangedDim.addAtIndex(angle, horiz, vert, orig.getAtIndex(horiz, vert, angle));
					//}
					
				}
			}
		}
		return gridChangedDim;
		
	}
	
	public static ComplexGrid3D fourierAngle(ComplexGrid3D projections, Config conf ){
		Fourier ft = new Fourier();
		ft.fft(projections, 0);
		projections.setSpacing(conf.getAngleIncrement(), conf.getPixelXSpace(), conf.getPixelYSpace());
		
//		for (int horz = 0; horz < projections.getSize()[1]; horz++){
//			for(int vert = 0; vert < projections.getSize()[2]; vert++){
//				for(int angle = 0; angle < projections.getSize()[0]; angle++){
//					
//				}
//			}
//		}
		return projections;
	}
	private static ComplexGrid2D centerAndShift(ComplexGrid2D orig, Config conf){
		int height = orig.getSize()[1];
		int width = orig.getSize()[0];
		ComplexGrid2D result = new ComplexGrid2D(width, height);
		for(int wuCounter = 0; wuCounter < height; wuCounter++){
			 
			for(int kArrayPos = 0; kArrayPos < width; kArrayPos++){
//				float kPositionF = (conf.kSpacing.getAtIndex(kArrayPos));
//				int kPositionI= (int)(kPositionF);
//				if(kPositionI < 0){
//					System.out.println("kArrayPos: "+ kArrayPos + ", kPositionI:" +kPositionI);
//				}
				//result.setAtIndex((int)(conf.kSpacing.getAtIndex(kArrayPos)+width/2), wuCounter, orig.getAtIndex(kArrayPos, wuCounter));
			//result.multiplyAtIndex((int)(conf.kSpacing.getAtIndex(kArrayPos) + width/2), wuCounter, conf.frequencyShift.getAtIndex(kArrayPos));
			}
		}
		
		return result;
	}


}

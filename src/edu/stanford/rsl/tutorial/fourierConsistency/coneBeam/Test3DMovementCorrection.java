package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class Test3DMovementCorrection {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		
		new ImageJ();
		Configuration.loadConfiguration();
		// read in projection data
		Grid3D projections = null;
		ApodizationImageFilter filt = new ApodizationImageFilter();
		filt.setCustomSizes(new Integer[]{null, (int)(Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight()-40)});
		try {
			// locate the file
			// here we only want to select files ending with ".bin". This will open them as "Dennerlein" format.
			// Any other ImageJ compatible file type is also OK.
			// new formats can be added to HandleExtraFileTypes.java
			//String filenameString = FileUtil.myFileChoose("proj/ciptmp/co98jaha/workspace/data/FinalProjections80kev/FORBILD_Head_80kev.tif",".tif", false);
			// call the ImageJ routine to open the image:
			//ImagePlus imp = IJ.openImage("D:/Data/ForbildHead3D/FinalProjections80kev/TestXCAT.tif");
			ImagePlus imp = IJ.openImage("D:/Data/WeightBearing/PMB/XCatDynamicSquat_NoTruncation_256proj_620_480_MeadianRad2_80keVnoNoise.tif");
			// Convert from ImageJ to Grid3D. Note that no data is copied here. The ImageJ container is only wrapped. Changes to the Grid will also affect the ImageJ ImagePlus.
			projections = ImageUtil.wrapImagePlus(imp);
			filt.configure();
			// Display the data that was read from the file.
			//projections.show("Data from file");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//NumericPointwiseOperators.fill(projections,0f);
		//projections.setAtIndex(0, 0, 0, 1);
		Grid3D projectionsWindowed = (Grid3D) ImageUtil.applyFilterInParallel((Grid3D) projections.clone(), filt).clone();
		projectionsWindowed.show("Windowed Projections");
		
		// get configuration
		String xmlFilename = "D:/Data/WeightBearing/PMB/XCAT motion correction_256proj_620_480_Subj2 Static60_fullView.xml";
		String outXMLfile = xmlFilename;
		outXMLfile = outXMLfile.substring(0, outXMLfile.length()-4);
		outXMLfile += "corrected.xml";
		
		Config conf = new Config(xmlFilename, 2, 1);
		conf.getMask().show("mask");
		System.out.println("N: "+ conf.getHorizontalDim() + " M: " + conf.getVerticalDim() + " K: "+  conf.getNumberOfProjections());
		
		
		MovementCorrection3D mc = new MotionCorrection3DFast(projectionsWindowed, conf, false);
		mc.doFFT2();
		mc.transposeData();

		Grid1D result = mc.computeOptimalShift();
		result.show("Result shift vector");
		
		Grid3D reconNoCorr = (Grid3D)ImageUtil.applyFiltersInParallel((Grid3D)projections.clone(), Configuration.getGlobalConfiguration().getFilterPipeline()).clone();
		reconNoCorr.show("Non-corrected Reconstruction");
		
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		for (int i = 0; i < geom.getProjectionMatrices().length; i++) {
			int idx = i*2;
			SimpleMatrix mat = SimpleMatrix.I_3.clone();
			mat.setElementValue(0, 2, -result.getAtIndex(idx)/mc.getConfig().getPixelXSpace());
			mat.setElementValue(1, 2, -result.getAtIndex(idx+1)/mc.getConfig().getPixelYSpace());
			
			geom.getProjectionMatrix(i).setKValue(
					SimpleOperators.multiplyMatrixProd(mat, geom.getProjectionMatrix(i).getK()));
		}
		Configuration.getGlobalConfiguration().setGeometry(geom);
		
		Grid3D recon = (Grid3D) ImageUtil.applyFiltersInParallel((Grid3D)projections.clone(), Configuration.getGlobalConfiguration().getFilterPipeline()).clone();
		recon.show("Corrected Reconstruction");
		
		mc.backTransposeData();
//		//mc.getData().show("2D-Fouriertransformed after transposing");
		mc.doiFFT2();
		mc.getData().getRealGrid().show("Output (corrected) - Real");
		mc.getData().getImagGrid().show("Output (corrected) - Imaginary");
		//mc.getData().show("Output (corrected) - Magnitude");

		Configuration.saveConfiguration(Configuration.getGlobalConfiguration(), outXMLfile);
		
	}
	
}

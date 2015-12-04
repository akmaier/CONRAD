/*
 /*
 * Copyright (C) 2015 Wolfgang Aichinger, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators.boundaryHandling;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.RamLakRampFilter;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.DoubleFunction;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.fourierConsistency.coneBeam.ApodizationImageFilter.windowType;

public class Test3DMovementCorrection {

	/**
	 * @param args
	 */
	public static void main(String[] args) {


		new ImageJ();
		
		//String xmlFilenameOptimization = "D:/Data/ForbildHead3D/ConradSettingsForbild3D_rescaledAnglesSmall.xml";
		//String xmlFilenameRealProjData = "D:/Data/ForbildHead3D/ConradSettingsForbild3D.xml";
		
		String xmlFilenameOptimization = "D:/Data/WeightBearing/PMB/XCAT motion correction_256proj_620_480_Subj2 Static60_fullView.xml";
		String xmlFilenameRealProjData = "D:/Data/WeightBearing/PMB/XCAT motion correction_256proj_620_480_Subj2 Static60_fullView.xml";
		
		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(xmlFilenameOptimization));
		// read in projection data
		Grid3D projectionsForOptimization = null;
		Grid3D projectionsRealProjData = null;
		ApodizationImageFilter filt = new ApodizationImageFilter();
		filt.setCustomSizes(new Integer[]{null, (int)(Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight()), null});
		filt.setWtU(windowType.rect);
		filt.setWtV(windowType.blackman);
		filt.setWtK(windowType.rect);
		
		RampFilteringTool rft = new RampFilteringTool();
		RamLakRampFilter ramLak = new RamLakRampFilter();

		try {
			// locate the file
			// here we only want to select files ending with ".bin". This will open them as "Dennerlein" format.
			// Any other ImageJ compatible file type is also OK.
			// new formats can be added to HandleExtraFileTypes.java
			//String filenameString = FileUtil.myFileChoose("proj/ciptmp/co98jaha/workspace/data/FinalProjections80kev/FORBILD_Head_80kev.tif",".tif", false);
			// call the ImageJ routine to open the image:
			
			//ImagePlus imp = IJ.openImage("D:/Data/ForbildHead3D/FinalProjections80kev/FORBILD_Head_80kev_Motion_rescaledToPowerOfTwoSmall.tif");
			//ImagePlus realProjData = IJ.openImage("D:/Data/ForbildHead3D/FinalProjections80kev/FORBILD_Head_80kev_Motion.tif");
			
			ImagePlus imp = IJ.openImage("D:/Data/WeightBearing/PMB/XCatDynamicSquat_NoTruncation_256proj_620_480_MeadianRad2_80keVnoNoise.tif");
			ImagePlus realProjData = IJ.openImage("D:/Data/WeightBearing/PMB/XCatDynamicSquat_NoTruncation_256proj_620_480_MeadianRad2_80keVnoNoise.tif");
			
			
			// Convert from ImageJ to Grid3D. Note that no data is copied here. The ImageJ container is only wrapped. Changes to the Grid will also affect the ImageJ ImagePlus.
			projectionsForOptimization = ImageUtil.wrapImagePlus(imp);
			projectionsRealProjData = ImageUtil.wrapImagePlus(realProjData);
			filt.configure();
			// Display the data that was read from the file.
			//projections.show("Data from file");

			ramLak.configure();
			rft.setRamp(ramLak);
			rft.setConfigured(true);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//NumericPointwiseOperators.fill(projections,0f);
		//projections.setAtIndex(0, 0, 0, 1);
		Grid3D projectionsWindowed = (Grid3D) ImageUtil.applyFiltersInParallel((Grid3D) projectionsForOptimization.clone(), new ImageFilteringTool[]{filt,rft}).clone();
		projectionsWindowed.show("Windowed Projections");

		// get configuration
		String outXMLfile = xmlFilenameRealProjData;
		outXMLfile = outXMLfile.substring(0, outXMLfile.length()-4);
		outXMLfile += "corrected.xml";

		//Config conf = new Config(xmlFilenameOptimization, 0, 1, 127.8, 100); // For Forbild Phantom
		Config conf = new Config(xmlFilenameOptimization, 0, 1, 168.9, 100); // For Forbild Phantom
		conf.getMask().show("mask");
		System.out.println("N: "+ conf.getHorizontalDim() + " M: " + conf.getVerticalDim() + " K: "+  conf.getNumberOfProjections());


		MovementCorrection3D mc = new MotionCorrection3DFast(projectionsWindowed, conf, false);
		mc.doFFT2();
		mc.transposeData();

		Grid1D parVecResult = mc.computeOptimalShift();
		parVecResult.show("Result shift vector");

		Grid2D results = new Grid2D(parVecResult.getSize()[0]/2, 2);
		for (int i = 0; i < results.getSize()[0]; i++) {
			results.setAtIndex(i, 0, parVecResult.getAtIndex(i*2));
			results.setAtIndex(i, 1, parVecResult.getAtIndex(i*2+1));
		}
		results.setOrigin(0,0);
		results.setSpacing(Configuration.getGlobalConfiguration().getGeometry().getAverageAngularIncrement()*Math.PI/180, 1);
		 

		Configuration.setGlobalConfiguration(Configuration.loadConfiguration(xmlFilenameRealProjData));
		Grid3D reconNoCorr = (Grid3D)ImageUtil.applyFiltersInParallel((Grid3D)projectionsRealProjData.clone(), Configuration.getGlobalConfiguration().getFilterPipeline()).clone();
		reconNoCorr.show("Non-corrected Reconstruction");

		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		Trajectory newGeom = new Trajectory(geom);

		Grid2D reconResults = new Grid2D(geom.getNumProjectionMatrices(), 2);
		reconResults.setOrigin(0,0);
		reconResults.setSpacing(geom.getAverageAngularIncrement()*Math.PI/180,1);
		for (int i = 0; i < reconResults.getSize()[0]; i++) {
			double[] recWC1 = reconResults.indexToPhysical(i, 0);
			double[] recWC2 = reconResults.indexToPhysical(i, 1);
			double[] optIdx1 = results.physicalToIndex(recWC1[0],recWC1[1]);
			double[] optIdx2 = results.physicalToIndex(recWC2[0],recWC2[1]);
			reconResults.setAtIndex(i, 0, InterpolationOperators.interpolateLinear(results,optIdx1[0],optIdx1[1],boundaryHandling.REPLICATE));
			reconResults.setAtIndex(i, 1, InterpolationOperators.interpolateLinear(results,optIdx2[0],optIdx2[1],boundaryHandling.REPLICATE));
		}
		for (int i = 0; i < reconResults.getHeight(); i++) {
			reconResults.getSubGrid(i).show("Interpolated shifts t" + i);
		}

		for (int i = 0; i < newGeom.getProjectionMatrices().length; i++) {
			SimpleMatrix mat = SimpleMatrix.I_3.clone();
			mat.setElementValue(0, 2, -reconResults.getAtIndex(i,0)/geom.getPixelDimensionX());
			mat.setElementValue(1, 2, -reconResults.getAtIndex(i,1)/geom.getPixelDimensionY());

			newGeom.setProjectionMatrix(i, new Projection(
					SimpleOperators.multiplyMatrixProd(mat, newGeom.getProjectionMatrix(i).computeP())));
		}
		Configuration.getGlobalConfiguration().setGeometry(newGeom);

		Grid3D recon1 = (Grid3D) ImageUtil.applyFiltersInParallel((Grid3D)projectionsRealProjData.clone(), Configuration.getGlobalConfiguration().getFilterPipeline()).clone();
		recon1.show("Corrected Reconstruction (No Freq Removed)");

		
		
		
		
		
		newGeom = new Trajectory(geom);
		
		ComplexGrid2D cplxResults = new ComplexGrid2D(results);
		Fourier fft = new Fourier();
		fft.fft(cplxResults);
		ComplexGrid2D subtractionLine = new ComplexGrid2D(cplxResults);
		for (int i = 2; i < subtractionLine.getSize()[0]/2; i++) {
			subtractionLine.setAtIndex(i, 0, 0);
			subtractionLine.setAtIndex(subtractionLine.getSize()[0]-i, 0, 0);
			subtractionLine.setAtIndex(i, 1, 0);
			subtractionLine.setAtIndex(subtractionLine.getSize()[0]-i, 1, 0);
		}
		fft.ifft(subtractionLine);
		for (int i = 0; i < subtractionLine.getSize()[1]; i++) {
			subtractionLine.getSubGrid(i).getRealGrid().show("Subtracted low-frequ. shifts t" + i);
		}		
		cplxResults.setAtIndex(0, 0, 0);cplxResults.setAtIndex(0, 1, 0);
		cplxResults.setAtIndex(1, 0, 0);cplxResults.setAtIndex(1, 1, 0);
		cplxResults.setAtIndex(cplxResults.getSize()[0]-1, 0, 0); cplxResults.setAtIndex(cplxResults.getSize()[0]-1, 1, 0);
		fft.ifft(cplxResults);
		results = cplxResults.getRealGrid();
		results.setOrigin(cplxResults.getOrigin());
		results.setSpacing(cplxResults.getSpacing());

		reconResults = new Grid2D(geom.getNumProjectionMatrices(), 2);
		reconResults.setOrigin(0,0);
		reconResults.setSpacing(geom.getAverageAngularIncrement()*Math.PI/180,1);
		for (int i = 0; i < reconResults.getSize()[0]; i++) {
			double[] recWC1 = reconResults.indexToPhysical(i, 0);
			double[] recWC2 = reconResults.indexToPhysical(i, 1);
			double[] optIdx1 = results.physicalToIndex(recWC1[0],recWC1[1]);
			double[] optIdx2 = results.physicalToIndex(recWC2[0],recWC2[1]);
			reconResults.setAtIndex(i, 0, InterpolationOperators.interpolateLinear(results,optIdx1[0],optIdx1[1],boundaryHandling.REPLICATE));
			reconResults.setAtIndex(i, 1, InterpolationOperators.interpolateLinear(results,optIdx2[0],optIdx2[1],boundaryHandling.REPLICATE));
		}
		for (int i = 0; i < reconResults.getHeight(); i++) {
			reconResults.getSubGrid(i).show("Interpolated shifts t" + i);
		}

		for (int i = 0; i < newGeom.getProjectionMatrices().length; i++) {
			SimpleMatrix mat = SimpleMatrix.I_3.clone();
			mat.setElementValue(0, 2, -reconResults.getAtIndex(i,0)/geom.getPixelDimensionX());
			mat.setElementValue(1, 2, -reconResults.getAtIndex(i,1)/geom.getPixelDimensionY());

			newGeom.setProjectionMatrix(i, new Projection(
					SimpleOperators.multiplyMatrixProd(mat, newGeom.getProjectionMatrix(i).computeP())));
		}
		Configuration.getGlobalConfiguration().setGeometry(newGeom);
		
		Grid3D recon2 = (Grid3D) ImageUtil.applyFiltersInParallel((Grid3D)projectionsRealProjData.clone(), Configuration.getGlobalConfiguration().getFilterPipeline()).clone();
		recon2.show("Corrected Reconstruction (Low Freq Removed)");

		mc.backTransposeData();
		//		//mc.getData().show("2D-Fouriertransformed after transposing");
		mc.doiFFT2();
		mc.getData().getRealGrid().show("Output (corrected) - Real");
		mc.getData().getImagGrid().show("Output (corrected) - Imaginary");
		//mc.getData().show("Output (corrected) - Magnitude");

		Configuration.saveConfiguration(Configuration.getGlobalConfiguration(), outXMLfile);

	}

}

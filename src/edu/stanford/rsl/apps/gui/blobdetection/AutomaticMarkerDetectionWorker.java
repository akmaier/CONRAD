package edu.stanford.rsl.apps.gui.blobdetection;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.io.Opener;
import ij.measure.Calibration;
import ij.process.AutoThresholder;
import ij.process.BinaryProcessor;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import ij.process.StackStatistics;

import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.CosineWeightingTool;
import edu.stanford.rsl.conrad.filtering.ExtremeValueTruncationFilter;
import edu.stanford.rsl.conrad.filtering.ImageConstantMathFilter;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.ImagePlusProjectionDataSource;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLBackProjector;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.motion.compensation.OpenCLCompensatedBackProjectorTPS;


public class AutomaticMarkerDetectionWorker extends MarkerDetectionWorker{

	/*
	 * This class determines the 3D positions of circular beads. First the beads are detected
	 *  using a Fast Radial Symmetry transform on the projection images. The outcome is then
	 *  low-pass filtered and backprojected to the 3D domain. After thresholding in 3D the 
	 *  beads are detected using a connected components analysis and centroids are extracted.
	 *  
	 */


	int nrOfBeads = -1;

	boolean showRefBeadsReconstruction = false;

	Roi cropRoi = null;

	public AutomaticMarkerDetectionWorker(){
		super();
	}


	@Override
	public void configure() {
		config = Configuration.getGlobalConfiguration();
		if (image==null)
			image = IJ.getImage();
		configured = true;
		if (nrOfBeads > 0){
			initializeMarkerPositions(nrOfBeads);
		}
		else{
			fastRadialSymmetrySpace = FRST();
			Grid3D frst = new Grid3D(fastRadialSymmetrySpace);
			initializeMarkerPositions(frst, false);
		}
		update2Dreference();
		configured = true;
	}


	public void blankOutMarkerPositions(Grid3D frst){
		//remove markers from FRST result
		for (int i = 0; i < twoDPosReal.size(); i++) {
			for (int j = 0; j < twoDPosReal.get(i).size(); j++) {
				double uv[] = twoDPosReal.get(i).get(j);
				int blankRadius = (int)Math.ceil(DoubleArrayUtil.minAndMaxOfArray(radiusOfBeads)[1]);
				for (int u = (int)Math.floor(uv[0])-blankRadius; u < (int)Math.ceil(uv[0])+blankRadius; ++u){
					for (int v = (int)Math.floor(uv[1])-blankRadius; v < (int)Math.ceil(uv[1])+blankRadius; ++v){
						if (u >= 0 && v >= 0 && u <= frst.getSize()[0] && v <= frst.getSize()[1])
							frst.setAtIndex(u, v, (int)uv[2], 0f);
					}
				}
			}
		}
	}


	public void initialize3DMarkerPositionsOnly(VOIBasedReconstructionFilter customBackprojector){
		fastRadialSymmetrySpace = FRST();
		Grid3D frst = new Grid3D(fastRadialSymmetrySpace);
		initializeMarkerPositions(frst, false, customBackprojector);
	}

	private void initializeMarkerPositions(int nrOfMarkers){
		fastRadialSymmetrySpace = FRST();
		if (cropRoi!=null){
			ImagePlus ip = ImageUtil.wrapGrid3D(fastRadialSymmetrySpace, "");
			ip.setRoi(cropRoi);
			IJ.run(ip, "Multiply...", "value=0 stack");
		}
		Grid3D frst = new Grid3D(fastRadialSymmetrySpace);
		initializeMarkerPositions(frst, false);
		update2Dreference();
		int oldSize = threeDPos.size()-1;
		while (threeDPos.size() < nrOfMarkers && oldSize==threeDPos.size()-1){
			oldSize = threeDPos.size();
			run();
			blankOutMarkerPositions(frst);
			//frst.show();
			// add only global maximum position of reconstruction
			initializeMarkerPositions(frst, true);
			update2Dreference();
		}
	}


	private int[] findMaximumOfStack(Grid3D input){
		float max = Float.NEGATIVE_INFINITY;
		int[] tmp = new int[3];
		for (int i = 0; i < input.getSize()[2]; i++) {
			for (int j = 0; j < input.getSize()[1]; j++) {
				for (int k = 0; k < input.getSize()[0]; k++) {
					if (input.getAtIndex(k, j, i) > max){
						max = input.getAtIndex(k, j, i);
						tmp[0]=k;
						tmp[1]=j;
						tmp[2]=i;
					}
				}
			}
		}
		return tmp;
	}


	private void initializeMarkerPositions(Grid3D frst, boolean findMaximumOnly){
		initializeMarkerPositions(frst, findMaximumOnly, null);
	}

	private void initializeMarkerPositions(Grid3D frst, boolean findMaximumOnly, VOIBasedReconstructionFilter customBackprojector){
		// Calculate the FRST -> subtract threshold -> set minimum to 0 ->  apply 2D Gauss -> backproject

		Grid3D	frstIn = new Grid3D(frst);

		CosineWeightingTool cwt = new CosineWeightingTool();
		cwt.configure();

		ImageConstantMathFilter imf = new ImageConstantMathFilter();
		imf.setOperation(" subtract ");		
		imf.setOperand(this.binarizationThreshold);
		imf.setConfigured(true);

		ExtremeValueTruncationFilter evtf = new ExtremeValueTruncationFilter();
		evtf.setOperation(" min ");
		evtf.setThreshold(0);
		evtf.setConfigured(true);

		Grid3D tmp = doParallelStuff(frstIn, new ImageFilteringTool[] {cwt, imf, evtf});
		// Apply 2D Gauss filter to remove high frequencies for the backprojection
		/*Filtering2DTool filt2D = new Filtering2DTool();
		filt2D.setFilter2D(ImageUtil.create2DGauss(9, DoubleArrayUtil.computeMean(this.radiusOfBeads)));
		filt2D.setConfigured(true);*/
		IJ.run(ImageUtil.wrapGrid3D(tmp, "Gaussian Input"), "Gaussian Blur...","sigma="+(2*DoubleArrayUtil.computeMean(this.radiusOfBeads)) + " stack");

		// Use redundancy weights as we might have a short scan
		// ParkerWeightingTool pwt = new TrajectoryParkerWeightingTool();
		// Apply backprojection (no ramp filtering required as we want to have the beads smeared anyway)

		VOIBasedReconstructionFilter oclb = null;
		if (customBackprojector != null)
			oclb = customBackprojector;
		else
			oclb = new OpenCLBackProjector();
		
		try {
			//pwt.configure();
			oclb.configure();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (oclb instanceof OpenCLCompensatedBackProjectorTPS){
			ImageGridBuffer igb = new ImageGridBuffer();
			igb.set(tmp);
			((OpenCLCompensatedBackProjectorTPS)oclb).setInitialRigidTransform(SimpleMatrix.I_4.clone());
			oclb.setShowStatus(true);
			try {
				((OpenCLCompensatedBackProjectorTPS)oclb).loadInputQueue(igb);
			} catch (IOException e) {
				e.printStackTrace();
			}
			tmp = ((OpenCLCompensatedBackProjectorTPS)oclb).reconstructCL();
		}
		else
			tmp = doParallelStuff(tmp, new ImageFilteringTool[] {oclb});



		// Use Maximum Entropy thresholding technique in 3D to find a suitable threshold
		ImagePlus ip = ImageUtil.wrapGrid3D(tmp, "Bead Reconstruction");
		if (showRefBeadsReconstruction)
			ip.show();
		IJ.run(ip, "Gaussian Blur 3D...", "x=5 y=5 z=5");

		//ip.getProcessor().setAutoThreshold("MaxEntropy dark stack");
		//IJ.run("Convert to Mask", "method=MaxEntropy background=Default black");

		if ( !findMaximumOnly ){
			StackStatistics sst = new StackStatistics(ip);
			AutoThresholder at = new AutoThresholder();
			int th = at.getThreshold("MaxEntropy", sst.histogram);
			double threshold = th*sst.binSize;
			AutomaticMarkerDetectionWorker.thresholdStack(ip, threshold);

			Calibration calibrationNew = new Calibration();
			calibrationNew.xOrigin = config.getGeometry().getOriginInPixelsX();
			calibrationNew.yOrigin = config.getGeometry().getOriginInPixelsY();
			calibrationNew.zOrigin = config.getGeometry().getOriginInPixelsZ();
			calibrationNew.pixelWidth = config.getGeometry().getReconVoxelSizes()[0];
			calibrationNew.pixelHeight = config.getGeometry().getReconVoxelSizes()[1];
			calibrationNew.pixelDepth = config.getGeometry().getReconVoxelSizes()[2];
			ip.setCalibration(calibrationNew);

			// Find the connected components and their centroids in 3D
			ConnectedComponent3D pc = new ConnectedComponent3D();
			pc.setLabelMethod(ConnectedComponent3D.MAPPED);
			// get the particles and do the analysis
			//final long start = System.nanoTime();
			double[] minMax = DoubleArrayUtil.minAndMaxOfArray(this.radiusOfBeads);
			double maxBeadSize = 4*Math.PI*Math.pow(minMax[1],3)*calibrationNew.pixelWidth*calibrationNew.pixelHeight*calibrationNew.pixelDepth/3;
			double minBeadSize = 4*Math.PI*Math.pow(minMax[0],3)*calibrationNew.pixelWidth*calibrationNew.pixelHeight*calibrationNew.pixelDepth/3;

			Object[] result = pc.getParticles(ip, 4, 0.01*minBeadSize, 1e6*maxBeadSize,
					ConnectedComponent3D.FORE, false);
			// calculate particle labelling time in ms
			//final long time = (System.nanoTime() - start) / 1000000;
			//IJ.log("Particle labelling finished in "+time+" ms");

			int[][] particleLabels = (int[][]) result[1];
			long[] particleSizes = pc.getParticleSizes(particleLabels);
			double[] volumes = pc.getVolumes(ip, particleSizes);
			double[][] centroids = pc.getCentroids(ip, particleLabels, particleSizes);

			SimpleVector sub = new SimpleVector(config.getGeometry().getReconVoxelSizes());
			sub.multiplyElementWiseBy(new SimpleVector(config.getGeometry().getOriginInPixelsX(),config.getGeometry().getOriginInPixelsY(),config.getGeometry().getOriginInPixelsZ()));

			// create new threeDPos at beginning otherwise add to existing one
			if (threeDPos == null){
				this.threeDPos = new ArrayList<double[]>(centroids.length);
			}
			for (int i = 1; i < volumes.length; i++) {
				if (volumes[i] > 0) {
					SimpleVector center = new SimpleVector(centroids[i]);
					center.subtract(sub);
					threeDPos.add(center.copyAsDoubleArray());
				}
			}
		}
		else{
			// create new threeDPos at beginning otherwise add to existing one
			if (threeDPos == null){
				this.threeDPos = new ArrayList<double[]>();
			}
			SimpleVector sub = new SimpleVector(config.getGeometry().getReconVoxelSizes());
			sub.multiplyElementWiseBy(new SimpleVector(config.getGeometry().getOriginInPixelsX(),config.getGeometry().getOriginInPixelsY(),config.getGeometry().getOriginInPixelsZ()));

			int[] pos = findMaximumOfStack(tmp);
			SimpleVector center = new SimpleVector(pos[0],pos[1],pos[2]);
			center.multiplyElementWiseBy(new SimpleVector(config.getGeometry().getReconVoxelSizes()));
			center.subtract(sub);
			threeDPos.add(center.copyAsDoubleArray());
		}
	}




	/**
	 * Threshold stack and create binary masks
	 * 
	 * @param ip the ImagePlus
	 * @param offset the threshold
	 */
	public static void thresholdStack(ImagePlus ip, double offset){

		ImageStack is = new ImageStack(ip.getWidth(), ip.getHeight(), ip.getStackSize());
		for (int slice = 1; slice <= ip.getStackSize(); ++slice){
			ImageProcessor img = ip.getStack().getProcessor(slice);
			byte[] pixels = new byte[img.getWidth()*img.getHeight()];
			ImageProcessor imp = new BinaryProcessor(new ByteProcessor(img.getWidth(), img.getHeight(), pixels));
			is.setProcessor(imp, slice);
			for (int j = 0; j< img.getHeight(); j++){
				for (int i = 0; i< img.getWidth(); i++){
					if (img.getPixelValue(i, j) > offset) {
						imp.set(i,j,255);
					}
				}
			}
		}
		ip.setStack(is);
	}


	public static Grid3D doParallelStuff(Grid3D inputStack, ImageFilteringTool[] filters){
		// run all the filters in parallel on the slices
		Grid3D outputStack=null;
		try {
			ImagePlusDataSink sink = new ImagePlusDataSink();
			sink.configure();
			ImagePlusProjectionDataSource pSource = new ImagePlusProjectionDataSource();
			pSource.setImage(inputStack);
			ParallelImageFilterPipeliner filteringPipeline = new ParallelImageFilterPipeliner(pSource, filters, sink);
			filteringPipeline.project();
			outputStack = sink.getResult();
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return outputStack;

	}


	public void setParameters(double binThresh, double circularity, double lowGradThresh, double distance, String radii, boolean refinementActive, int nrOfBeads){
		super.setParameters(binThresh, circularity, lowGradThresh, distance, radii);

		if (refinementActive == false && this.nrOfBeads != -1){
			this.nrOfBeads = -1;
			configured = false;
		}
		if (refinementActive == true && this.nrOfBeads == -1)
		{
			this.nrOfBeads = nrOfBeads;
			configured = false;
		}
		if (refinementActive == true && this.nrOfBeads != -1)
		{
			if (this.nrOfBeads != nrOfBeads){
				this.nrOfBeads = nrOfBeads;
				configured = false;
			}
		}
	}

	public void setShowRefBeadsReconstruction(boolean showRefBeadsReconstruction) {
		this.showRefBeadsReconstruction = showRefBeadsReconstruction;
	}

	public void setCropRoi(Roi cropRoi) {
		this.cropRoi = cropRoi;
	}

	public Roi getCropRoi() {
		return cropRoi;
	}


	public static void main(String[] args) {
		CONRAD.setup();

		String dir = "D:\\Data\\WeightBearing\\stanford_knee_data_jang_2013_07_08\\Weight-bearing_NO6_IR1\\Projection\\AfterPreCorrection\\";
		/*File folder = new File(dir);
		File[] listOfFiles = folder.listFiles(); 

		for (int i = 0; i < listOfFiles.length; i++) 
		{
			if (listOfFiles[i].isFile()) 
			{
				files = listOfFiles[i].getName();
				if (files.endsWith(".tif"))
				{
					Opener op = new Opener();
					//op.openImage("C:\\Users\\berger\\Desktop\\Result of WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy-2.tif");
					//ImagePlus ip = op.openImage("D:\\Data\\WeightBearing\\stanford_knee_data_jang_2013_07_08\\Weight-bearing_NO6_IR1\\Projection\\AfterPreCorrection\\WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy.tif");
					//ImagePlus ip = op.openImage("D:\\Data\\WeightBearing\\stanford_knee_data_jang_2013_07_08\\Weight-bearing_NO6_IR1\\Projection\\AfterPreCorrection\\WEIGHT-BEARING.XA._.11.Standing_Straight_1Leg255Lbs_AdditionalWeight2Cokes_248views_70kV_0.54uGy_ObjectOutOfFOV.tif");
					ImagePlus ip = op.openImage(dir + files);
					ip.setTitle(files);
					MarkerDetectionWorker mdw = new AutomaticMarkerDetectionWorker();
					mdw.setImage(ip);
					mdw.setParameters(0.2, 3, 0, 25, "[3]");
					ip.show();
					mdw.run();
				}
			}
		}
		 */
		Opener op = new Opener();
		ImagePlus ip = op.openImage(dir + "WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy.tif");
		ip.setTitle("WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy.tif");
		MarkerDetectionWorker mdw = new AutomaticMarkerDetectionWorker();
		mdw.setImage(ip);
		((AutomaticMarkerDetectionWorker)mdw).setParameters(0.5, 3, 0, 25, "[3]", true, 10);
		ip.show();
		mdw.run();

		/*ImagePlus ip = op.openImage("C:\\Users\\berger\\Desktop\\Bead Reconstruction.tif");
		Configuration config = Configuration.getGlobalConfiguration();
		Calibration calibrationNew = new Calibration();
		calibrationNew.xOrigin = config.getGeometry().getOriginInPixelsX();
		calibrationNew.yOrigin = config.getGeometry().getOriginInPixelsY();
		calibrationNew.zOrigin = config.getGeometry().getOriginInPixelsZ();
		calibrationNew.pixelWidth = config.getGeometry().getReconVoxelSizes()[0];
		calibrationNew.pixelHeight = config.getGeometry().getReconVoxelSizes()[1];
		calibrationNew.pixelDepth = config.getGeometry().getReconVoxelSizes()[2];
		ip.setCalibration(calibrationNew);

		// Find the connected components and their centroids in 3D
		ConnectedComponent3D pc = new ConnectedComponent3D();
		pc.setLabelMethod(ConnectedComponent3D.MAPPED);
		// get the particles and do the analysis
		final long start = System.nanoTime();

		Object[] result = pc.getParticles(ip, 4, 0, Double.POSITIVE_INFINITY,
				ConnectedComponent3D.FORE, false);
		// calculate particle labelling time in ms
		final long time = (System.nanoTime() - start) / 1000000;
		IJ.log("Particle labelling finished in "+time+" ms");

		int[][] particleLabels = (int[][]) result[1];
		long[] particleSizes = pc.getParticleSizes(particleLabels);
		final int nParticles = particleSizes.length;
		double[] volumes = pc.getVolumes(ip, particleSizes);
		double[][] centroids = pc.getCentroids(ip, particleLabels, particleSizes);
		int[][] limits = pc.getParticleLimits(ip, particleLabels, nParticles);

		String units = ip.getCalibration().getUnit();
		ResultsTable rt = new ResultsTable();
		for (int i = 1; i < volumes.length; i++) {
			if (volumes[i] > 0) {
				rt.incrementCounter();
				rt.addLabel(ip.getTitle());
				rt.addValue("ID", i);
				rt.addValue("Vol. (" + units + "�)", volumes[i]);
				rt.addValue("x Cent (" + units + ")", centroids[i][0]);
				rt.addValue("y Cent (" + units + ")", centroids[i][1]);
				rt.addValue("z Cent (" + units + ")", centroids[i][2]);
				rt.updateResults();
			}
		}
		rt.show("Results");*/

	}
}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

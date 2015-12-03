/* Copyright (C) 2015 Bernhard Stimpel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.StringTokenizer;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.ResultsTable;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.StackStatistics;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.CosineWeightingTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.SheppLoganRampFilter;
import edu.stanford.rsl.conrad.filtering.redundancy.ParkerWeightingTool;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.pipeline.BufferedProjectionSink;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.MotionCompensatedRecon;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.ECG.ECGGating;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.mip.OpenCLMaximumIntensityProjection;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.morphology.Morphology;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.morphology.StructuringElement;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.reconWithStreakReduction.ConeBeamBackprojectorStreakReduction;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.reconWithStreakReduction.ConeBeamBackprojectorStreakReductionWithMotionCompensation;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.registration.bUnwarpJ_.MiscTools;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.registration.bUnwarpJ_.Param;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.registration.bUnwarpJ_.Transformation;
import edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.registration.bUnwarpJ_.bUnwarpJ_;



/**
 * Class to model Cardiac Vasculature Reconstruction with Residual Motion 
 * Compensation using ECG gating, Backprojection with Streak Reduction
 * Forward MIP and MotionCompensated Reconstruction
 * */
public class MotionCompensatedRecon{
	
	//	TODO Set ROI dimensions if needed
	//	x and y define the top left corner of the ROI.  		
	private int xROI = 0;
	private int yROI = 0;
	private int widthROI = 960;
	private int heightROI = 960;	
	
	public Grid3D transformationVectorsX = null;
	public Grid3D transformationVectorsY = null;
		
	// Projection geometry parameters
	private double sizeU = 960;
	private double sizeV = 960;
	private double spacingU = 0.32;
	private double spacingV = 0.32;
	private double sourceDetectorDist = 1200;
	private double sourceIsocenterDist = 800;
	
	private int gat_ign = 3;	
	
	// ECG Parameters
	private String ecgFile = "C:/workspace/CavarevData/phasedata_card.txt";
	static double REF_HEART_PHASE = 0.90;	
	private double gatingWidth = 4.0;
	private double gatingSlope = 4.0;
	
	Configuration cnfg = null;

	// Path to the ecg and pmat files
	private String projMatricesFile = "C:/workspace/CavarevData/cavarev.matrices.bin";
	static String projectionFile = "C:/workspace/CavarevData/cavarev_noBreathing.tif";
	static String registrationPath = "C:/workspace/CavarevData/Registration/";

	public MotionCompensatedRecon(){	
	}
	
	public static void main(String args[]){
		
		new ImageJ();
		
		//	Computing the initial reconstruction
		MotionCompensatedRecon moCoRecon = new MotionCompensatedRecon();		
		Grid3D initialReco = null;		
		try {
			System.out.println("Starting the initial reconstruction.");
			initialReco = moCoRecon.reconstructConeBeam(false, moCoRecon.getGat_ign());		
			} catch (Exception e) {
			e.printStackTrace();
		}

		Grid3D resultingReco = moCoRecon.IterativeSteps(initialReco);
		resultingReco.show();
	}
			
	
	/**
	 * Steps that can be repeated iteratively during Cardiac Vasculature Reconstruction
	 * @param resultRecon: result of reconstruction (in first iteration result of initial reconstruction in next iteration result of MotionCompReconstruction)
	 * @param projectionsPath: Path where the projections are saved
	 * @param iterationCounter: used to save image with corresponding Number of Iteration
	 * @param referenceHeartPhase: ECG gating parameter
	 * @param numHeartBeats: ECG gating parameter
	 * */
	public Grid3D IterativeSteps(Grid3D resultRecon){		
		
		Grid3D source = null;
		Grid3D target = null;
		configure();
		
		// Preparing the source and target images for the registration process
		System.out.println("Preparing the initial reconstruction.");
		target = prepReconstruction(resultRecon);

		System.out.println("Pre-processing the original projections.");
		source = preProcessProj();
		
		transformationVectorsX = new Grid3D(source.getSize()[0], source.getSize()[1], source.getSize()[2]);
		transformationVectorsY = new Grid3D(target.getSize()[0], target.getSize()[1], target.getSize()[2]);
		
		// Cropping the images to save computation time.
		source = cropImage(source, xROI, yROI, widthROI, heightROI);	
		target = cropImage(target, xROI, yROI, widthROI, heightROI);
		
		bUnwarpJ_ bUnwarpJ = new bUnwarpJ_();

		//	Starting the registration process. The resulting deformationfields are saved in the transformationVectors. 
		performRegistration(target, source, bUnwarpJ);
		
		//	Performing the reconstruction.
		resultRecon = reconstructConeBeam(true, getGat_ign());
		
		return resultRecon;
	}
	
	/**
	 * This method performs a percentile histogram thresholding followed by a Maximum Intensity Forward Projection.
	 * @param initReconstruction: the initial reconstruction
	 * @return the Maximum Intensity Forward Projection as Grid3D
	 */
	public Grid3D prepReconstruction(Grid3D initReconstruction){
		ImagePlus reconIP = ImageUtil.wrapGrid3D(initReconstruction, "");
		
		//TODO Adjusting the thresholding values in this method may be necessary 
		
		//	Adapt thresholding value to current image intensity values via percentile thresholding based on histograms
		double percentile = 0.001;	
		Grid3D reconGrid = percentileHistogramThreshold(reconIP, percentile);
		
		//	Perform the maximum intensity projection
		Grid3D mipProjGrid = mipProjection(reconGrid);
		
		//	Second thresholding operation
		Grid3D fwpCopy = threshold3D(mipProjGrid, 0.011);
		return fwpCopy;
	}
	
	/** This methods performs the pre-processing of the original projections. 
	 * In the first step a top-hat filter is applied to the projections. The necessary parameters
	 * are defined in the CoronaryConfig. In the second step the achieved result is binarized.  
	 * @return the filtered and binarized projections as Grid3D
 	 */	
	public Grid3D preProcessProj(){		
		//TODO	Adjusting the filter size and binarization threshold in this method may be necessary
		
		//Performing the morphological filtering
		ImagePlus preProjIP = morphFilterProjs(projectionFile, 10, "topHat");
		
		//Wrapping and binarizing the morph-filtered projections
		Grid3D preProjGrid = ImageUtil.wrapImagePlus(preProjIP);			
		preProjGrid = binarize3D(preProjGrid, 3.0);		
		
		return preProjGrid;
	}
	
	/**
	 * ConeBeamReconstruction Method using ConeBeamBackprojector from Tutorial Code
	 * including: ConeBeamCosineFilter, Parker Weights and RamLakFilter
	 * */
	public Grid3D reconstructConeBeam(boolean motionCompensation, int gat_ign){
		
		configure();		
		Grid3D sino = null;
		Grid3D resultRecon = null;		

		// Preparing the projections in advance to the reconstruction. 
		try {
			sino = prepProjections();
		} catch (Exception e) {
			e.printStackTrace();
		}		
		
		//	Performing the actual backprojection, either with or without motion compensation. 
		if(motionCompensation == true){
			ConeBeamBackprojectorStreakReductionWithMotionCompensation cbbp = new ConeBeamBackprojectorStreakReductionWithMotionCompensation();
			cbbp.setGat_ign(gat_ign);
			resultRecon = cbbp.backprojectPixelDrivenCL(sino, transformationVectorsX, transformationVectorsY);
		}
		else{
			ConeBeamBackprojectorStreakReduction cbbp = new ConeBeamBackprojectorStreakReduction();
			cbbp.setGat_ign(gat_ign);		
			resultRecon = cbbp.backprojectPixelDrivenCL(sino);
		}
		return resultRecon;
	}
	
	/**
	 * Method performing the Maximum Intensity Projection
	 * @param inputProjection: The pre-processed inputProjection as Grid3D
	 * @return the resulting Maximum Intensity Forward Projection as Grid3D
	 */
	public Grid3D mipProjection(Grid3D inputProjection){
		ImagePlus resultingMIP = null;
		
		//perform the actual Maximum Intensity Projection
		OpenCLMaximumIntensityProjection clForwardProjector = new OpenCLMaximumIntensityProjection();		
		try {
			clForwardProjector.configure();
		} catch (Exception e) {
			e.printStackTrace();
		}				
		clForwardProjector.setTex3D(ImageUtil.wrapGrid3D(inputProjection, "Maximum Intensity Forward Projection"));
		resultingMIP = clForwardProjector.project();
		
		//Wraping and returning the result
		Grid3D resultingMIPGrid = ImageUtil.wrapImagePlus(resultingMIP);
		return resultingMIPGrid;
	}
	
	/**
	 * Preparing the original projections for reconstruction.
	 * @param projections : Original projections (currently not used; projectionstream as source)
	 * @return The prepared projections 
	 * @throws Exception
	 */
	private Grid3D prepProjections() throws Exception{
		
		Trajectory geo  = cnfg.getGeometry();
		
		//	Initializing and configuring the tools. 
		CosineWeightingTool cbTool = new CosineWeightingTool();
		cbTool.configure();		
		ParkerWeightingTool redundancyTool = new ParkerWeightingTool(geo);
		redundancyTool.configure();		
		RampFilteringTool rampFiltTool = new RampFilteringTool();		
		RampFilter ramp = new SheppLoganRampFilter();		
		rampFiltTool.setRamp(ramp);
		rampFiltTool.setConfigured(true);
		
		ImageFilteringTool[] filters = new ImageFilteringTool[]{cbTool,redundancyTool,rampFiltTool};
		
		//	Retrieving the projection source as projection stream.
		ProjectionSource pSource = FileProjectionSource.openProjectionStream(projectionFile);
		BufferedProjectionSink sink = new ImagePlusDataSink();
		
		//	Creating the filter pipeline. 
		ParallelImageFilterPipeliner filteringPipeline = 
				new ParallelImageFilterPipeliner(pSource, filters, sink);
				
		//	Executing the previously created pipeline.
		Thread thread = new Thread(new Runnable(){
			public void run(){
				try {
					filteringPipeline.project();
				} catch (Exception e) {
					e.printStackTrace();
				}
			} 
		});
		thread.start();
		
		//	Applying the ECG-gating to the projection source 
		Grid3D preppedProj = sink.getResult();
		double[] ecg = readEcg(ecgFile);
		preppedProj = applyGating(preppedProj, ecg);
		
		return preppedProj;
	}
	
	/**
	 * Method to perform image registration using bUnwarpJ
	 * Gets the targetGrid, sourceGrid and an instance of bUnwarpJ
	 * the transformationVectors in 2D in x and y direction are saved 
	 * and can be accessed via getters, the registration windows are
	 * closed after each registration of mapping source to target
	 * */
	public void performRegistration(Grid3D targetGrid, Grid3D sourceGrid, bUnwarpJ_ bUnwarpJ){
		
		bUnwarpJ_.setRichOutput(true);
		bUnwarpJ_.setSaveTransformation(true);	
		
		// Iterate over all projection slices
		for(int i = 0; i < targetGrid.getSize()[2]; i++){
			
				System.out.println("Perform registration for slice: " + i);
			
				// Take the current slice from the whole stack and convert it to the ImagePlus format
				Grid2D targetGridSlice = targetGrid.getSubGrid(i);
				Grid2D sourceGridSlice = sourceGrid.getSubGrid(i);				
				
				ImageProcessor target = ImageUtil.wrapGrid2D(targetGridSlice);
				ImageProcessor source = ImageUtil.wrapGrid2D(sourceGridSlice);
				
				ImagePlus ipTarget = new ImagePlus("Target", target);
				ImagePlus ipSource = new ImagePlus("Source", source);				
				
				//TODO Defining the Registration Parameters
				// - accuracy mode (0 - Fast, 1 - Accurate, 2 - Mono)
				// - minimum scale deformation (0 - Very Coarse, 1 - Coarse, 2 - Fine, 3 - Very Fine)
				// - maximum scale deformation (0 - Very Coarse, 1 - Coarse, 2 - Fine, 3 - Very Fine, 4 - Super Fine)
				//  - divergence weight
				// - curl weight
				// - landmark weight
				// - similarity weight
				// - consistency weight
				// - stopping threshold			
				Param parameters = new Param(2, 0 ,3, 4, 0.5, 0.5, 0, 1, 10, 0.01);
				
				//	Starting the actual registration process
				bUnwarpJ_.alignImagesBatch(ipTarget, ipSource, null, null, parameters);
				
				//	Get transformation vectors from GrayscaleResultTileMaker
				double[][] transformation_x = Transformation.getTrafo_x();
				double[][] transformation_y = Transformation.getTrafo_y();				
				
				if(i < targetGrid.getSize()[2]){
					final double transformedImage [][] = new double [transformation_x.length][transformation_x[0].length];
					int stepv = Math.min(Math.max(10, transformation_x.length/15),30);
					int stepu = Math.min(Math.max(10, transformation_x[0].length/15),30);
	
					for (int v=0; v<transformation_x.length; v++){
						for (int u=0; u<transformation_x[0].length; u++){
							transformedImage[v][u]=255;
						}
					}
					
					for (int v=0; v< transformation_x.length; v+=stepv){
						for (int u=0; u< transformation_x[0].length; u+=stepu){
								final double x = transformation_x[v][u];
								final double y = transformation_y[v][u];
									MiscTools.drawArrow(
											transformedImage,
											u,v,(int)Math.round(x),(int)Math.round(y),0,2);
						}
					}
	
					// Set it to the image stack
					FloatProcessor fp=new FloatProcessor(transformation_x[0].length,transformation_x.length);
					for (int v=0; v<transformation_x.length; v++){
						for (int u=0; u<transformation_x[0].length; u++){
							fp.putPixelValue(u, v, transformedImage[v][u]);
						}
					}					

					//Grid2D grid = ImageUtil.wrapFloatProcessor(fp);
					//grid.show("DeformationVectors");
				}				
				
				//	Compute real deformation vectors				
				for(int v = 0; v < transformation_x.length; v ++){
					for(int u = 0; u < transformation_x[0].length; u++){
						
						final double x = transformation_x[v][u];
						final double y = transformation_y[v][u];
						transformation_x[v][u] = (x - u);
						transformation_y[v][u] = (y - v);						
					}
				}
				Grid2D grid2D_x = convertToGrid2D(transformation_x);
				Grid2D grid2D_y = convertToGrid2D(transformation_y);				
			
				this.transformationVectorsX.setSubGrid(i, grid2D_x);
				this.transformationVectorsY.setSubGrid(i, grid2D_y);
								
				IJ.run("Close All");
		}		
	 }

	 /**
	  * Method to configure the geometry
	  */
	public void configure(){
		
		//	Loading the CONRAD config. 
		Configuration.loadConfiguration();
		Configuration cnfg = Configuration.getGlobalConfiguration();
		
		//	Reading the projection matrices.
		Trajectory geo = cnfg.getGeometry();
		Projection[] pMat = readProjections(projMatricesFile);
		double[] angles = new double[pMat.length];
		for(int i = 0; i < pMat.length; i++){
			angles[i] = 200.0d/(pMat.length-1)*i;
		}
		//	Loading the actual geometry. 
		geo.setProjectionMatrices(pMat);
		geo.setNumProjectionMatrices(pMat.length);
		geo.setProjectionStackSize(pMat.length);
		geo.setPrimaryAngleArray(angles);
		geo.setSourceToAxisDistance(sourceIsocenterDist);
		geo.setSourceToDetectorDistance(sourceDetectorDist);
		geo.setDetectorWidth((int)sizeU);
		geo.setDetectorHeight((int)sizeV);
		geo.setPixelDimensionX(spacingU);
		geo.setPixelDimensionY(spacingV);
		cnfg.setGeometry(geo);
		Configuration.setGlobalConfiguration(cnfg);
		this.cnfg = cnfg;
	}	
		
	/**convert array[][] to Grid2D*/
	public static Grid2D convertToGrid2D(double[][] array){

		int height = array.length;
		int width = array[0].length;
		
		Grid2D grid2D = new Grid2D(height, width);
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++){
				double val = array[i][j];
				grid2D.setAtIndex(i, j,(float) val);
			}
		}
		return grid2D;
	}
	
	
	/**Thresholding Method for a Grid3D.
	 * Values above threshold are kept, values below are set to zero.*/
	public Grid3D threshold3D(Grid3D grid, double threshold){
			int[] size = grid.getSize();
			int width = size[0];
			int height = size[1];
			int depth = size[2];
			Grid3D result = new Grid3D(width, height, depth);
			
			for(int i = 0; i < width; i ++){
				for(int j = 0; j < height; j++){
					for(int k = 0; k < depth; k++){
						if(grid.getAtIndex(i, j, k) < threshold){
							result.setAtIndex(i, j, k, 0.0f);
						}
						else{
							result.setAtIndex(i,j,k,grid.getAtIndex(i, j, k));
						}
					}
				}				
			}
			return result;
	}
	

	/**Binarization Method for a Grid3D. 
	 * values above threshold are set to one, values below to zero.*/
	public Grid3D binarize3D(Grid3D grid, double threshold){
			int[] size = grid.getSize();
			int width = size[0];
			int height = size[1];
			int depth = size[2];
			Grid3D result = new Grid3D(width, height, depth);
			
			for(int i = 0; i < width; i ++){
				for(int j = 0; j < height; j++){
					for(int k = 0; k < depth; k++){
						if(grid.getAtIndex(i, j, k) < threshold){
							result.setAtIndex(i, j, k, 0.0f);
						}
						else{
							result.setAtIndex(i, j, k, 1.0f);
						}
					}
				}				
			}
			return result;
	}
	
	/**threshold with percentile threshold*/
	public Grid3D percentileHistogramThreshold(ImagePlus imp, double percentile){
		
		//	Computing the threshold.
		long percentileNum = getPercentileNum(imp, percentile);
		double threshold = findThreshold(imp, percentileNum);
		
		// Performing the actual thresholding with the previously computed threshold value. 
		Grid3D grid = ImageUtil.wrapImagePlus(imp);
		grid = threshold3D(grid, threshold);
		
		return grid;
	}
	
	public long getPercentileNum(ImagePlus imp, double percentile){
		ResultsTable rt = new ResultsTable();
		long[] histogram = null; //is.histogram;
		long totalCount = 0;
		double value = 0.0;
		double binWidth = 0.0;
		double min = imp.getProcessor().getMin();
		StackStatistics ss1 = new StackStatistics(imp);
		min = ss1.histMin;
		if(imp.getBitDepth() == 8 || imp.getBitDepth() == 24){
			for(int i = 0; i < histogram.length; i++){
				rt.setValue("Value", i, i);
			    rt.setValue("Count", i, histogram[i]);
			}
		}else{
			value = min;
			binWidth = ss1.binSize;
			histogram = ss1.getHistogram();
			
			for (int i=0; i<histogram.length; i++) {
		     rt.setValue("Value", i, value);
		     rt.setValue("Count", i, histogram[i]);
		     value += binWidth;
		     totalCount += histogram[i];
			}
	
		}
		long percentileNum = (long) (percentile*totalCount);
		
		return percentileNum;
	}
	
	public double findThreshold(ImagePlus imp, long percentileNum){
		ResultsTable rt = new ResultsTable();
		long[] histogram = null; //is.histogram;
		long totalCount = 0;
		double value = 0.0;
		double binWidth = 0.0;
		StackStatistics ss1 = new StackStatistics(imp);
		double max = ss1.histMax;
		histogram = ss1.getHistogram();
		if(imp.getBitDepth() == 8 || imp.getBitDepth() == 24){
			for(int i = histogram.length - 1; i > - 1; i--){
				rt.setValue("Value", i, i);
			    rt.setValue("Count", i, histogram[i]);
			     if(percentileNum < totalCount){
			    	 return value;
			     }
			}
		}else{
			value = max;
			binWidth = ss1.binSize;
			histogram = ss1.getHistogram();
			
			for(int i = histogram.length - 1; i > - 1; i--){
		     value -= binWidth;
		     totalCount += histogram[i];
		     if(percentileNum < totalCount){
		    	 return value;
		     }
			}
			rt.show("Threshold Results");
		}
		return 0.0;
	}
	/**
	 * A method to perform a morphological filtering of the projections.  
	 * @param projFile : location of the projection file on the disc
	 * @param structElementSize : radius of the structuring element
	 * @param filter : choose which filter should be used. Currently "topHat" or "dilation" are implemented. 
	 * @return ImagePlus with the filtered result.
	 */
	public static ImagePlus morphFilterProjs(String projFile, int structElementSize, String filter){
		
		//	Creating the structuring element.
		StructuringElement s = new StructuringElement("Circle", structElementSize, false);		
		
		// Loading the projection source and creating an ImageStack.
		Grid3D projGrid = ImageUtil.wrapImagePlus(IJ.openImage(projFile));		
		ImagePlus result = null;		
		ImageStack stack = new ImageStack(projGrid.getSize()[0],projGrid.getSize()[1]);
		
		// Applying the morphological filter slice by slice. 
		System.out.println("Applying the morphological filter. This may take a while...");
		for(int i = 0; i < projGrid.getSize()[2]; i++)
		{			
			ImagePlus workPlus = new ImagePlus("Img", ImageUtil.wrapGrid2D(projGrid.getSubGrid(i)));
			if(filter == "topHat") result = Morphology.topHat(workPlus, s, true);
			if(filter == "dilation") result = Morphology.dilate(workPlus, s);
			stack.addSlice(result.getProcessor());			
		}
		
		ImagePlus projIP = new ImagePlus("ProjectionsFiltered", stack);
		return projIP;
	}
	
	
	public int getGat_ign() {
		return gat_ign;
	}

	public void setGat_ign(int gat_ign) {
		this.gat_ign = gat_ign;
	}
	
	private Projection[] readProjections(String filename){
		 
		Projection[] pMat = null;
		try {
			FileInputStream fStream = new FileInputStream(filename);
			// Number of matrices is given as the total size of the file
			// divided by 4 bytes per float, divided by 12 floats per projection matrix
			int nMat = (int) (fStream.getChannel().size() / 4 / (4*3));
			DataInputStream in = new DataInputStream(fStream);
			
			pMat = new Projection[nMat];
			
			for(int m = 0; m < nMat; m++){
				SimpleMatrix mat = new SimpleMatrix(3,4);
				for(int i = 0; i < mat.getRows(); i++){
					for(int j = 0; j < mat.getCols(); j++){
						byte[] buffer = new byte[4];
					    int bytesRead = in.read(buffer);
					    float val = 0;
					    if(bytesRead == 4){
					    	val = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getFloat();
					    }
					    mat.setElementValue(i, j, val);
					}
				}
				pMat[m] = new Projection(mat);
			}
			
			in.close();
			fStream.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return pMat;
	}
	
	/**
	 * Reads the ECG from a file. It is assumed that there are as many heart phases in a separate line
 	 * as there are projections.
	 * @param filename
	 * @return
	 */
	private double[] readEcg(String filename){
		ArrayList<Double> e = new ArrayList<Double>();
		FileReader fr;
		try {
			fr = new FileReader(filename);		
			BufferedReader br = new BufferedReader(fr);			
			String line = br.readLine();
			while(line != null){
				StringTokenizer tok = new StringTokenizer(line);
				e.add(Double.parseDouble(tok.nextToken()));
				line = br.readLine();
			}
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		double[] ecg = new double[e.size()];
		for(int i = 0; i < e.size(); i++){
			ecg[i] = e.get(i);
		}
		return ecg;
	}	
	/**
	 * Uses the ECGGating class to perform cosine-function based gating of the acquisition.
	 * @param img
	 * @param ecg
	 * @return
	 */
	private Grid3D applyGating(Grid3D img, double[] ecg){
		ECGGating gating = new ECGGating(gatingWidth, gatingSlope, REF_HEART_PHASE);
		return gating.weightProjections(img, ecg);
	}
	/**
	 * Returns the registration save path.
	 */
	public static String getRegistrationPath(){
		return registrationPath;
	}
	
	/**
	 * This method "crops" the image to the specified roi.
	 * The image doesn't get cropped actually, but all values outside the roi are set to zero.
	 */
	public Grid3D cropImage(Grid3D inputGrid, int x, int y, int width, int height ){
		
		for(int k = 0; k < inputGrid.getSize()[2]; k++){
			for(int i = 0; i < inputGrid.getSize()[0]; i++){
				for(int j = 0; j < inputGrid.getSize()[1]; j++){
					if(i < y || i > y + height || j < x || j > x+width){						
							inputGrid.setAtIndex(i, j, k, 0.0f);
					}
				}
			}
		}		
		return inputGrid;
	}
	
}


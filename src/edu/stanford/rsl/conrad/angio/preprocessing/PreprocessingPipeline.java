package edu.stanford.rsl.conrad.angio.preprocessing;

import java.io.File;
import java.util.ArrayList;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.Dijkstra2D;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.MinimumSpanningTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;
import edu.stanford.rsl.conrad.angio.motion.Gating;
import edu.stanford.rsl.conrad.angio.points.DistanceTransform2D;
import edu.stanford.rsl.conrad.angio.points.DistanceTransformUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.noise.NoiseFiltering;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.gradient.Koller2D;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.Frangi2D;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.Sato2D;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.ExtractConnectedComponents;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.HysteresisThresholding;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.image.HistogramPercentile;
import edu.stanford.rsl.conrad.angio.util.image.ImageOps;
import edu.stanford.rsl.conrad.angio.util.io.EcgIO;
import edu.stanford.rsl.conrad.angio.util.io.EdgeListIO;
import edu.stanford.rsl.conrad.angio.util.io.PointAndRadiusIO;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;
import edu.stanford.rsl.tutorial.iterative.GridOp;

public class PreprocessingPipeline {

	protected DataSet dataset = null;
	protected Angiogram completeAngiogram;
	protected Angiogram gatedAngiogram;
	protected Angiogram preprocessedAngiogram;
	
	private double refHeartPhase = 0.9;
	
	String outDir = null;
	
	protected boolean initialized = false;
	private boolean writeOutput = false;
	
	public static void main(String[] args) {
		DataSets datasets = DataSets.getInstance();
		
		int caseID = 3;		
		DataSet ds = datasets.getCase(caseID);
		String outputDir = ds.getDir()+ "eval/";
		
		double refHeartPhase = -1;
		
		PreprocessingPipeline prepPipe = new PreprocessingPipeline(ds, outputDir);
		prepPipe.setWriteOutput(true);
		prepPipe.setRefHeartPhase(refHeartPhase);
		prepPipe.evaluate();
		Angiogram ang = prepPipe.getPreprocessedAngiogram();
		
		new ImageJ();
		ang.getProjections().show();
		ang.getReconCostMap().show();
	}
	
	
	public PreprocessingPipeline(DataSet ds){
		this.dataset = ds;
	}
	
	public PreprocessingPipeline(DataSet ds, String outDir){
		this.dataset = ds;
		this.outDir = outDir;
	}
	
	public void evaluate(){
		String dir = this.outDir;
		boolean readEcgFlag = false;
		if(refHeartPhase<0){
			dir += "all/";
			readEcgFlag = true;
		}else{
			dir += String.valueOf(refHeartPhase)+"/";
		}
		File fTest = new File(dir);
		if(fTest.exists()){
			System.out.println("Reading from folder: "+dir);
			Projection[] ps = ProjMatIO.readProjMats(dir+"pMat.txt");
			Grid3D cm = ImageUtil.wrapImagePlus(IJ.openImage(dir+"distTrafo.tif"));
			Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(dir+"img.tif"));
			Angiogram prepAng = new Angiogram(img, ps, new double[ps.length]);
			if(readEcgFlag){
				double[] ecg = EcgIO.readEcg(dataset.getPreproSet().getEcgFile());
				prepAng.setEcg(ecg);
			}
			prepAng.setReconCostMap(cm);
			ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
			Grid3D binImg = SkeletonUtil.costMapToVesselTreeImage(cm);
			for(int i = 0; i < ps.length; i++){
				skels.add(SkeletonUtil.binaryImgToSkel(binImg.getSubGrid(i), 0, false));
			}
			prepAng.setSkeletons(skels);
			this.preprocessedAngiogram = prepAng;
		}else{
			System.out.println("Starting the preprocessing.");
			
			if(!initialized){
				init();
			}
			
			Angiogram prep = new Angiogram(gatedAngiogram);
			// noise filtering
			prep.setProjections(noiseSuppression(prep.getProjections()));
			// vessel segmentation, centerline extraction, distance transform
			prep.setSkeletons(centerlineExtractionPipeline(prep.getProjections()));
			prep.setReconCostMap(DistanceTransformUtil.slicewiseDistanceTransform(prep.getProjections(), prep.getSkeletons()));
			
			this.preprocessedAngiogram = prep;
		}
		if(isWriteOutput() && outDir != null){
			File f = new File(dir);
			if(!f.exists()){
				f.mkdirs();
			}
			IJ.saveAsTiff(preprocessedAngiogram.getProjectionsAsImP(), dir+"img.tif");
			IJ.saveAsTiff(ImageUtil.wrapGrid3D(preprocessedAngiogram.getReconCostMap(),""), dir+"distTrafo.tif");
			ProjMatIO.writeProjTable(preprocessedAngiogram.getPMatrices(), dir+"pMat.txt");
		}
		System.out.println("All done.");
	}


	protected ArrayList<Skeleton> centerlineExtractionPipeline(Grid3D img){
				
		Grid3D tubeEnhanced = vesselEnhancement(img);
		Grid3D centEnhanced = centerlineEnhancement(img);
		
		Grid3D combined = new Grid3D(centEnhanced);
		NumericPointwiseOperators.multiplyBy(combined, tubeEnhanced);
		ArrayList<Skeleton> skeletons = extractCenterline(combined);
		return skeletons;
	}
		
	protected ArrayList<Skeleton> extractCenterline(Grid3D img){
		ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
		Grid3D threshImg = new Grid3D(img);
		Grid3D vtImg = new Grid3D(img);
		for(int k = 0; k < img.getSize()[2]; k++){
			System.out.println("Centerline extraction on slice "+String.valueOf(k+1)+ " of " + String.valueOf(img.getSize()[2])+".");			
			// Threshold the vessel enhanced image to get a subset of centerline candidates and remove isolated pixels
			Grid2D thresh = ImageOps.thresholdImage(img.getSubGrid(k), dataset.getPreproSet().getCostMapThreshold());
			ExtractConnectedComponents conComp = new ExtractConnectedComponents();
			conComp.setShowResults(false);
			conComp.setDilationSize(0);
			conComp.setMinimumPointsInRegionDouble(6);
			thresh = conComp.runSlice(thresh);
			thresh = ImageOps.thresholdImage(thresh,0.0); // set values to 0-1 after connected comp.
			threshImg.setSubGrid(k, thresh);
			// extract centerline candidates
			ArrayList<PointND> pts = ImageOps.thresholdedPointList(thresh, 0.5);
			// extract largest component
			MinimumSpanningTree mst = new MinimumSpanningTree(
					pts, dataset.getPreproSet().getDijkstraMaxDistance()/thresh.getSpacing()[0]);
			mst.run();
			ArrayList<ArrayList<PointND>> mstCC = mst.getConnectedComponents();
			ArrayList<ArrayList<PointND>> largestComp;
			largestComp = mst.getLargestComponents(mstCC, dataset.getPreproSet().getNumLargestComponents());			
			Grid2D vtFin = new Grid2D(thresh.getSize()[0],thresh.getSize()[1]);
			vtFin.setSpacing(thresh.getSpacing());
			vtFin.setOrigin(thresh.getOrigin());
			for(int i = 0; i < largestComp.size(); i++){
				Grid2D vtImage = mst.visualize(thresh, largestComp.get(i));
				// now skeletonize using centerline extraction
				// determine start point as point in largest connected region
				Point startPoint = SkeletonUtil.determineStartPoint(largestComp.get(i));
				// determine end point candidates
				ArrayList<Point> endPts = new ArrayList<Point>();
				for(int j = 0; j < largestComp.get(i).size(); j++){
					Point ep = new Point((int)largestComp.get(i).get(j).get(0),(int)largestComp.get(i).get(j).get(1),0);
					endPts.add(ep);
				}
				// calculate distance transform from the centerline candidates
				DistanceTransform2D distTrafo = new DistanceTransform2D(vtImage, largestComp.get(i), true);
				distTrafo.setVerbose(false);
				Grid2D cm = distTrafo.run();
				distTrafo.unload();
				// combine with vessel enhancement measure to create local minima
				cm = combineCostMeasures(cm, img.getSubGrid(k));
				
				// extract allowed paths connecting end point candidates and start node
				Dijkstra2D dijkstra = new Dijkstra2D();
				dijkstra.setVerbose(false);
				dijkstra.setPruningLength(2.5);
				dijkstra.run(cm, cm, startPoint, dataset.getPreproSet().getDijkstraMaxDistance(), endPts);
				vtImage = dijkstra.visualizeVesselTree(dijkstra.getVesselTree());
				NumericPointwiseOperators.addBy(vtFin, vtImage);
				vtImg.setSubGrid(k, vtFin);
			}			
			Skeleton s = SkeletonUtil.binaryImgToSkel(vtFin, 0, false);
			skels.add(s);
		}
		
		return skels;
	}
	
	private Grid2D combineCostMeasures(Grid2D distMap, Grid2D localCost){
		Grid2D costMap = new Grid2D(distMap);
		Roi roi = dataset.getPreproSet().getRoi();
		for(int i = 0; i < costMap.getSize()[0]; i++){
			for(int j = 0; j < costMap.getSize()[1]; j++){
				if(roi.contains(i, j)){
					float val;
					if(j == roi.getBounds().y){
						val = (float)dataset.getPreproSet().getDijkstraMaxDistance()-0.5f;
					}else{
						val = (1-localCost.getAtIndex(i, j)) + distMap.getAtIndex(i, j);
					}
					costMap.setAtIndex(i, j, val);
				}
			}
		}
		return costMap;
	}
	
	/**
	 * Enhances the centerline of vessel-like structures using measures derived from the derivative perpendicular to 
	 * the structure's orientation. The orientation is derived from the Hessian. 
	 * @param img
	 * @return
	 */
	protected Grid3D centerlineEnhancement(Grid3D img){
		Koller2D koller = new Koller2D(img);
		koller.setScales(dataset.getPreproSet().getHessianScales());
		koller.setRoi(dataset.getPreproSet().getRoi());
		koller.evaluate();
		Grid3D cents = koller.getResult();
		
		for(int k = 0; k < img.getSize()[2]; k++){
			Grid2D slice = new Grid2D(cents.getSubGrid(k));
			// Hysteresis thresholding of Sato response
			HistogramPercentile histPerc = new HistogramPercentile(slice);
			histPerc.setMin(0.0f);
			histPerc.setRoi(dataset.getPreproSet().getRoi());
			double[] th = new double[2];
			th[0] = histPerc.getPercentile(dataset.getPreproSet().getCenterlinePercentile()[0]);
			th[1] = histPerc.getPercentile(dataset.getPreproSet().getCenterlinePercentile()[1]);
			HysteresisThresholding hyst = new HysteresisThresholding(th[1], th[0]);
			Grid2D hysted = hyst.run(slice);
			// Remove small connected components
			ExtractConnectedComponents conComp = new ExtractConnectedComponents();
			conComp.setShowResults(false);
			conComp.setDilationSize(2*dataset.getPreproSet().getConCompDilation());
			Grid2D hystCC = conComp.removeSmallConnectedComponentsSlice(hysted, dataset.getPreproSet().getConCompSize());
			Grid2D normalized = ImageOps.normalizeOutsideMask(cents.getSubGrid(k), hystCC, th[0], 0.0f, true);
			cents.setSubGrid(k, normalized);
		}
//		new ImageJ();
//		cents.show();
		return cents;
	}
	
	/**
	 * Enhances vessel-like structures using the measures derived from the Hessian matrix.
	 * Hysteresis thresholding of the vessel-enhanced image.
	 * Moreover, we remove small connected components and then normalize the response to [0,1] outside the mask.
	 * @param img
	 * @return
	 */
	public Grid3D vesselEnhancement(Grid3D img){		
		Frangi2D vness = new Frangi2D(img);
		vness.setScales(dataset.getPreproSet().getHessianScales());
		vness.setRoi(dataset.getPreproSet().getRoi());
		vness.setStructurenessPercentile(dataset.getPreproSet().getStructurenessPercentile());
		vness.evaluate(dataset.getPreproSet().getGammaThreshold());
		Grid3D vnessImg = vness.getResult();
		for(int k = 0; k < img.getSize()[2]; k++){
			Grid2D slice = new Grid2D(vnessImg.getSubGrid(k));
			// Hysteresis thresholding of Sato response
			HistogramPercentile histPerc = new HistogramPercentile(slice);
			double[] th = new double[2];
			th[0] = histPerc.getPercentile(dataset.getPreproSet().getVesselPercentile()[0]);
			th[1] = histPerc.getPercentile(dataset.getPreproSet().getVesselPercentile()[1]);
			HysteresisThresholding hyst = new HysteresisThresholding(th[1], th[0]);
			Grid2D hysted = hyst.run(slice);
			// Remove small connected components
			ExtractConnectedComponents conComp = new ExtractConnectedComponents();
			conComp.setShowResults(false);
			conComp.setDilationSize(dataset.getPreproSet().getConCompDilation());
			Grid2D hystCC = conComp.removeSmallConnectedComponentsSlice(hysted, dataset.getPreproSet().getConCompSize());
			Grid2D normalized = ImageOps.normalizeOutsideMask(vnessImg.getSubGrid(k), hystCC, th[0], 0.0f);
			vnessImg.setSubGrid(k, normalized);
		}
//		new ImageJ();
//		vnessImg.show();
		return vnessImg;
	}
	
	/**
	 * Enhances vessel-like structures using the measures derived from the Hessian matrix.
	 * Hysteresis thresholding of the vessel-enhanced image.
	 * Moreover, we remove small connected components and then normalize the response to [0,1] outside the mask.
	 * @param img
	 * @return
	 */
	private Grid3D vesselEnhancementSato(Grid3D img){
		// Hessian-based vessel enhancement
		Sato2D tubeness = new Sato2D(img);
		tubeness.setScales(dataset.getPreproSet().getHessianScales());
		tubeness.setRoi(dataset.getPreproSet().getRoi());
		tubeness.evaluate();
		Grid3D sato = tubeness.getResult();
		for(int k = 0; k < img.getSize()[2]; k++){
			Grid2D slice = new Grid2D(sato.getSubGrid(k));
			// Hysteresis thresholding of Sato response
			HistogramPercentile histPerc = new HistogramPercentile(slice);
			double[] th = new double[2];
			th[0] = histPerc.getPercentile(dataset.getPreproSet().getVesselPercentile()[0]);
			th[1] = histPerc.getPercentile(dataset.getPreproSet().getVesselPercentile()[1]);
			HysteresisThresholding hyst = new HysteresisThresholding(th[1], th[0]);
			Grid2D hysted = hyst.run(slice);
			// Remove small connected components
			ExtractConnectedComponents conComp = new ExtractConnectedComponents();
			conComp.setShowResults(false);
			conComp.setDilationSize(dataset.getPreproSet().getConCompDilation());
			Grid2D hystCC = conComp.removeSmallConnectedComponentsSlice(hysted, dataset.getPreproSet().getConCompSize());
			// Normalize Sato Response to [0,1] outside of mask
			Grid2D normalized = ImageOps.normalizeOutsideMask(sato.getSubGrid(k), hystCC, th[0], 0.0f);
			sato.setSubGrid(k, normalized);
		}
		
		return sato;
	}
	
	/**
	 * Applies bilateral filtering to the input data after gating.
	 * @param img
	 * @return
	 */
	protected Grid3D noiseSuppression(Grid3D img){
		return NoiseFiltering.bilateralFilter(img, dataset.getPreproSet().getBilatWidth(),
				dataset.getPreproSet().getBilatSigmaDomain(), dataset.getPreproSet().getBilatSigmaPhoto());
	}
	
	/**
	 * Initializes class members from the data set provided if not already done so. Is able to re-perform gating only
	 * if a different heart-phase has been set during calls from outside. 
	 */
	protected void init() {
		System.out.println("Initializing values for preprocessing...");
		if(completeAngiogram == null){
			readData();
		}
		this.gatedAngiogram = Gating.applyGating(completeAngiogram,
													refHeartPhase,
													dataset.getPreproSet().getEcgWidth(),
													dataset.getPreproSet().getEcgSlope(),
													dataset.getPreproSet().getEcgHardThreshold() );
		System.out.println("Done.");
	}

	private void readData(){
		ImagePlus imp = IJ.openImage(dataset.getProjectionFile());
		double[] ecg = EcgIO.readEcg(dataset.getPreproSet().getEcgFile());
		Projection[] pMat = ProjMatIO.readProjMats(dataset.getRecoSet().getPmatFile());
		this.completeAngiogram = new Angiogram(imp, pMat, ecg);
	}
	
	public double getRefHeartPhase() {
		return refHeartPhase;
	}


	public void setRefHeartPhase(double refHeartPhase) {
		this.refHeartPhase = refHeartPhase;
		initialized = false;
	}


	public Angiogram getGatedAngiogram() {
		return gatedAngiogram;
	}

	public Angiogram getCompleteAngiogram() {
		return completeAngiogram;
	}

	public Angiogram getPreprocessedAngiogram(){
		return preprocessedAngiogram;
	}


	public boolean isWriteOutput() {
		return writeOutput;
	}


	public void setWriteOutput(boolean writeOutput) {
		this.writeOutput = writeOutput;
	}
}

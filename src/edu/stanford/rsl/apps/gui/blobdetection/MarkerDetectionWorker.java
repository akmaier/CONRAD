package edu.stanford.rsl.apps.gui.blobdetection;

import java.awt.Color;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Set;





import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.FastRadialSymmetryTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.ImagePlusProjectionDataSource;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.utils.XmlUtils;


import ij.IJ;
import ij.ImagePlus;
import ij.gui.Overlay;
import ij.measure.Calibration;
import ij.process.StackStatistics;

public class MarkerDetectionWorker implements GUIConfigurable{

	/**
	 * Marker detection tool based on manually annotated initial marker positions.
	 * Use PointSelector tool for marker annotation. Or store points in the following order:
	 * 
	 * ArrayList<ArrayList<double[]>>> pointSets
	 * 
	 * Where the outer ArrayList is for the different sets of points and its size equals the 
	 * number of markers
	 * The inner ArrayList contains the manually annotated positions of the markers and the
	 * float[] array contains the actual coordinates, where the first values correspond to 
	 * the x and y coordinates and the third to the slice offset starting from 0 to the number
	 * of slices minus 1
	 * 
	 * @author Martin Berger, Jang-Hwan Choi
	 */


	//	For 1mm bead	
	protected double[] radiusOfBeads = new double[]{3};	
	protected double binarizationThreshold = 0.5; // 0.75		
	protected double lowGradThresh = 0;	
	protected double circularity = 3;
	protected double distance = 10;//20

	protected double maxThresh = binarizationThreshold;
	protected double minThresh = binarizationThreshold;
	protected double threshDec = 0.2;

	protected Grid3D fastRadialSymmetrySpace = null;

	protected HashMap<Integer,ArrayList<ArrayList<PointND>>> allDetectedBeads = null;

	protected ArrayList<double[]> threeDPos = null;

	protected ArrayList<ArrayList<double[]>> twoDPosReal = null;

	protected ArrayList<ArrayList<double[]>> twoDPosRef = null;
	
	protected ArrayList<ArrayList<double[]>> twoDPosMerged = null;

	protected Configuration config = null;

	public Configuration getConfig() {
		return config;
	}

	public void setConfig(Configuration config) {
		this.config = config;
	}

	protected boolean configured = false;

	protected ImagePlus image = null;

	private String filenamePriors = null;

	public MarkerDetectionWorker(){
		threeDPos = null;
		twoDPosReal = null;
		twoDPosRef = null;
		twoDPosMerged = null;
		config = null;
		configured = false;
		image = null;
		fastRadialSymmetrySpace = null;
		allDetectedBeads = null;
	}

	public MarkerDetectionWorker(ArrayList<ArrayList<double[]>> twoDInit){
		this();
		twoDPosReal = twoDInit;
	}


	protected void updateReferences(){
		// create/update the 3D reference positions from the 2D data
		update3Dreference();
		// create/update the 2D reference positions from the 3D data
		update2Dreference();
	}


	public void run() {
		//Grid3D tester = new Grid3D(input.getSize()[0],input.getSize()[1],input.getSize()[2],true); 
		if (!configured){
			try {
				configure();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		// overlay for drawing color circles and crosses
		Overlay ov = new Overlay();
		image.setOverlay(ov);

		// reorder reference points
		System.out.println("Bead Detection: Nr of reference beads (Input): " + twoDPosRef.size());
		// Order: SliceNr (Integer) -> BeadNr (Integer) -> BeadPosition (PointND)
		HashMap<Integer,HashMap<Integer, PointND>> refPoints = new HashMap<Integer, HashMap<Integer,PointND>>();
		for (int i = 0; i < twoDPosRef.size(); i++) {
			for (int j = 0; j < twoDPosRef.get(i).size(); j++) {
				int sliceNr = (int)twoDPosRef.get(i).get(j)[2];
				if (!refPoints.containsKey(sliceNr))
					refPoints.put(sliceNr, new HashMap<Integer,PointND>());
				refPoints.get(sliceNr).put(i, new PointND(twoDPosRef.get(i).get(j)[0],twoDPosRef.get(i).get(j)[1]));
			}
		}

		// initialize the twoDPosReal and twoDPosMerged array
		// this is only for initialization and memory allocation
		twoDPosReal = new ArrayList<ArrayList<double[]>>(twoDPosRef.size());
		twoDPosMerged = new ArrayList<ArrayList<double[]>>(twoDPosRef.size());
		Iterator<ArrayList<double[]>> it1 = twoDPosRef.iterator();
		while (it1.hasNext()) {
			twoDPosReal.add(new ArrayList<double[]>());
			twoDPosMerged.add(new ArrayList<double[]>());
			Iterator<double[]> it2 = it1.next().iterator();
			while (it2.hasNext()) {
				twoDPosReal.get(twoDPosReal.size()-1).add(null);
				twoDPosMerged.get(twoDPosMerged.size()-1).add(null);
				it2.next();
			}
		}

		// run the fast radial symmetry transform for all projections
		if (fastRadialSymmetrySpace == null)
			fastRadialSymmetrySpace = FRST();
		// precompute candidate points over all projections
		if (allDetectedBeads == null){
			allDetectedBeads = precomputeCandidatePoints(refPoints.keySet());
		}


		// loop over the slices (not necessarily ordered)
		Iterator<Integer> sliceIt = refPoints.keySet().iterator();
		while (sliceIt.hasNext()) {
			int slice = sliceIt.next();
			HashMap<Integer,PointND> innerMap = refPoints.get(slice);

			// find closest matching points and assign to each other -> first compute all distances
			// saves threshold, distance and distance matrix coordinates
			ArrayList<Entry<Double,int[]>> distances = new ArrayList<Entry<Double,int[]>>(10*allDetectedBeads.get(slice).size());

			// threshold loop: decrease binarization threshold if no bead is found
			for (int i = 0; i < allDetectedBeads.get(slice).size(); i++) {

				// the potential center points for this threshold
				ArrayList<PointND> detectedBeads = allDetectedBeads.get(slice).get(i);

				// loop over the beads in this slice
				Iterator<Integer> beadIt = innerMap.keySet().iterator();
				while (beadIt.hasNext()) {
					int bead = beadIt.next();
					PointND priorPos = innerMap.get(bead);

					// find closest matching points and assign to each other -> first compute all distances
					// saves distance and distance matrix coordinates
					// Determine the distances between reference and detected beads
					for (int j = 0; j < detectedBeads.size(); j++) {
						distances.add(new AbstractMap.SimpleEntry<Double,int[]>(
								// distance between reference and detected beads
								priorPos.euclideanDistance(detectedBeads.get(j)),
								// coordinates [threshold, reference bead No., detected  bead No.]
								new int[]{i,bead,j}
								));
					}

					// print the reference point onto the result images
					if (i==0)
						printXAtPoint(ov, priorPos, slice, 3);
				}
			}

			// all measurements done! --> Determine the winners
			Collections.sort(distances, new Comparator<Entry<Double,int[]>>() {
				@Override
				public int compare(Entry<Double, int[]> o1,	Entry<Double, int[]> o2) {
					return (o1.getKey() < o2.getKey()) ? -1 : ( (o1.getKey() > o2.getKey()) ? 1 : 0 );
				}
			}
					);

			// start with best pairs and "remove" them from the search space 
			Set<Integer> refSet = new HashSet<Integer>();
			Set<Integer> detSet = new HashSet<Integer>();
			Iterator<Entry<Double,int[]>> distIt = distances.iterator();
			while (distIt.hasNext()) {
				Entry<Double,int[]> curr = distIt.next();
				if (curr.getKey() <= this.distance){
					int thresh = curr.getValue()[0];
					int ref = curr.getValue()[1];
					int det = curr.getValue()[2];
					if (refSet.contains(ref) || detSet.contains(det))
						continue;

					// this is the actual detection!
					refSet.add(ref); detSet.add(det);
					double[] coords = allDetectedBeads.get(slice).get(thresh).get(det).getCoordinates();
					twoDPosReal.get(ref).set(slice, new double[] {coords[0], coords[1], slice});
					twoDPosMerged.get(ref).set(slice, new double[] {coords[0], coords[1], slice});
					VisualizationUtil.printCrossAtPoint(ov, new PointND(twoDPosReal.get(ref).get(slice)), slice, 4, Color.red);
					System.out.println("Slice: " + slice + "\t" + "Bead: " + ref + "\t\t" + twoDPosReal.get(ref).get(slice)[0] + "\t" + twoDPosReal.get(ref).get(slice)[1] + "\t");
				}
			}
			
			// We missed some beads in the detection process! Add reference positions to the merged 2D positions...
			if (refSet.size() < innerMap.keySet().size()){
				Iterator<Integer> beadNrIt = innerMap.keySet().iterator();
				while (beadNrIt.hasNext()) {
					int bead = beadNrIt.next();
					if (!refSet.contains(bead)){
						twoDPosMerged.get(bead).set(slice, new double[]{innerMap.get(bead).get(0), innerMap.get(bead).get(1), slice});
						VisualizationUtil.printCrossAtPoint(ov, innerMap.get(bead), slice, 4, Color.blue);
					}
				}
			}
		}

		// output the detections in FRST space
		/*
		ImagePlus dottedPlus = ImageUtil.wrapGrid3D(fastRadialSymmetrySpace, "FRST Space + Detection Results");
		dottedPlus.setOverlay(ov);
		dottedPlus.show();
		*/

		// remove not found beads
		for (int i = twoDPosReal.size()-1; i >= 0 ; --i) {
			while(twoDPosReal.get(i).remove(null));
			if (twoDPosReal.get(i).size()<=0)
				twoDPosReal.remove(i);
		}
		
		// sanity check for the merged array! None of these things should happen
		for (int i = twoDPosMerged.size()-1; i >= 0 ; --i) {
			while(twoDPosMerged.get(i).remove(null)){
				System.out.println("Merged array had missing element!!! Something went wrong!!");
			}
			if (twoDPosMerged.get(i).size()<=0){
				System.out.println("Merged array had missing slice!!! Something went wrong!!");
				twoDPosMerged.remove(i);
			}
				
		}

		// update the 2D and 3D reference positions
		updateReferences();

		System.out.println("Bead Detection: Nr of detected beads (Output): " + twoDPosReal.size());
	}
	
	public ArrayList<ArrayList<double[]>> getMergedTwoDPositions(){
		return twoDPosMerged;
	}
	

	public void printXAtPoint(Overlay ov, PointND pos, int slice, double crossSize){
		VisualizationUtil.printCrossAtPoint(ov, pos, slice, crossSize, Color.green);
	}


	/**
	 * Precompute center points of detected connected components. 
	 * The threshold for binarization varies from maxThresh to minThresh.
	 * @param allHoughBeads The list that stores the candidate points over multiple thresholds
	 * @param slice The slice index
	 * 
	 */
	protected HashMap<Integer,ArrayList<ArrayList<PointND>>> precomputeCandidatePoints(Set<Integer> sliceNumberSet){
		HashMap<Integer,ArrayList<ArrayList<PointND>>> out = new HashMap<Integer, ArrayList<ArrayList<PointND>>>(); 
		Iterator<Integer> sliceIt = sliceNumberSet.iterator();
		while (sliceIt.hasNext()) {
			int slice = sliceIt.next();
			ArrayList<ArrayList<PointND>> sliceRes = new ArrayList<ArrayList<PointND>>();
			for (double thresh = binarizationThreshold; thresh >= binarizationThreshold; thresh -= threshDec) {
				ImagePlus ip = new ImagePlus("Thresholded Bead Map",General.thresholdImage(
						ImageUtil.wrapGrid2D(fastRadialSymmetrySpace.getSubGrid(slice)), thresh));
				
				Calibration calibrationNew = new Calibration();
				calibrationNew.xOrigin = 0;
				calibrationNew.yOrigin = 0;
				calibrationNew.zOrigin = 0;
				calibrationNew.pixelWidth = 1;
				calibrationNew.pixelHeight = 1;
				calibrationNew.pixelDepth = 1;
				ip.setCalibration(calibrationNew);
				
				ConnectedComponent3D pc = new ConnectedComponent3D();
				pc.setLabelMethod(ConnectedComponent3D.MAPPED);
				Object[] result = pc.getParticles(ip, 1, 0, Double.POSITIVE_INFINITY,
						ConnectedComponent3D.FORE, false);

				int[][] particleLabels = (int[][]) result[1];
				long[] particleSizes = pc.getParticleSizes(particleLabels);
				double[] volumes = pc.getVolumes(ip, particleSizes);
				double[][] centroids = pc.getCentroids(ip, particleLabels, particleSizes);
				
				ArrayList<PointND> beads = new ArrayList<PointND>(volumes.length);
				for (int i = 1; i < volumes.length; i++) {
					if (volumes[i] > 0) {
						beads.add(new PointND(centroids[i]));
					}
				}
				sliceRes.add(beads);
			}
			out.put(slice, sliceRes);
		}
		return out;
	}

	protected Grid3D FRST(){
		// ***************************Calculate the FRST in parallel ***********************************
		FastRadialSymmetryTool frst = new FastRadialSymmetryTool();
		frst.setRadii(this.radiusOfBeads);
		frst.setAlpha(this.circularity);
		frst.setSmallGradientThreshold(this.lowGradThresh);
		frst.setConfigured(true);
		ImageFilteringTool [] filts = {frst};
		Grid3D out=null;
		try {
			ImagePlusDataSink sink = new ImagePlusDataSink();
			sink.configure();
			ImagePlusProjectionDataSource pSource = new ImagePlusProjectionDataSource();
			pSource.setImage(ImageUtil.wrapImagePlus(image));
			ParallelImageFilterPipeliner filteringPipeline = new ParallelImageFilterPipeliner(pSource, filts, sink);
			filteringPipeline.project();
			out = sink.getResult();
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		StackStatistics sst = new StackStatistics(ImageUtil.wrapGrid3D(out,""));	
		System.out.println("FRST --> Maximum : " + sst.histMax + " / Minimum: " + sst.histMin);

		return out;
	}


	protected void update2Dreference() {
		twoDPosRef = new ArrayList<ArrayList<double[]>>(threeDPos.size());
		Projection[] pMatrices = config.getGeometry().getProjectionMatrices();

		for (int i=0; i < threeDPos.size(); ++i){
			twoDPosRef.add(new ArrayList<double[]>(pMatrices.length));

			for (int j = 0; j < pMatrices.length; j++) {
				twoDPosRef.get(i).add(compute2Dfrom3D(threeDPos.get(i), pMatrices[j], j));
			}

		}
	}


	protected void update3Dreference() {
		threeDPos = new ArrayList<double[]>(twoDPosReal.size());
		for (int i=0; i < twoDPosReal.size(); ++i){
			threeDPos.add(compute3Dfrom2D(twoDPosReal.get(i), config.getGeometry().getProjectionMatrices()));
		}
	}


	private double [] compute2Dfrom3D(double [] point3D, Projection pMatrix, int slice){

		// Compute coordinates in projection data.
		SimpleVector homogeneousPoint = SimpleOperators.multiply(pMatrix.computeP(), new SimpleVector(point3D[0], point3D[1], point3D[2], 1));
		// Do forward projection to 2D coordinates
		double coordU = homogeneousPoint.getElement(0) / homogeneousPoint.getElement(2);
		double coordV = homogeneousPoint.getElement(1) / homogeneousPoint.getElement(2);

		return new double [] {coordU, coordV, (double)slice};
	}


	private double [] compute3Dfrom2D(ArrayList<double[]> twoDpointSet, Projection[] pMatrices) {

		// extract corresponding projection matrices and build linear system of equations
		// height of the overall system matrix
		int highDim = twoDpointSet.get(0).length;
		int lowDim = highDim-1;

		// height of the system matrix
		int M = twoDpointSet.size()*(lowDim);

		SimpleMatrix A = new SimpleMatrix(M,highDim);
		SimpleMatrix b = new SimpleMatrix(M,1);

		for (int i=0; i < twoDpointSet.size(); ++i){
			double[] point = twoDpointSet.get(i);
			int sliceNr = (int)twoDpointSet.get(i)[twoDpointSet.get(i).length-1];
			SimpleMatrix pMatrix = pMatrices[sliceNr].computeP();
			// A * x = b

			SimpleMatrix R = pMatrix.getSubMatrix(0,0,pMatrix.getRows(),pMatrix.getCols()-1);
			SimpleMatrix t = pMatrix.getSubMatrix(0,pMatrix.getCols()-1,pMatrix.getRows(),1);

			// extract the 2D point and transform to a matrix
			SimpleMatrix ui = (new SimpleVector(point)).getSubVec(0, point.length-1).transposed().transposed();

			SimpleMatrix Ai = SimpleOperators.multiplyMatrixProd(
					ui,
					// extract the last row of the projection matrix
					R.getRow(R.getRows()-1).transposed()
					);
			// subtract the multiplication result by the first rows of the projection matrix
			Ai.subtract(R.getSubMatrix(0, 0, R.getRows()-1, R.getCols()));

			SimpleMatrix bi = t.getSubMatrix(0, 0, t.getRows()-1,t.getCols());
			bi.subtract(ui.multipliedBy(pMatrix.getElement(pMatrix.getRows()-1, pMatrix.getCols()-1)));

			A.setSubMatrixValue(i*lowDim, 0, Ai);
			b.setSubMatrixValue(i*lowDim, 0, bi);
		}

		// Compute the 3D point using the pseudoinverse
		SimpleVector X = Solvers.solveLinearLeastSquares(A, b.getCol(0));

		DecompositionSVD svd = new DecompositionSVD(A);
		System.out.println("Solving for 3D point using " + (M/lowDim) + " 2D points\nCondition number was: " + svd.cond());
		return X.copyAsDoubleArray();
	}


	public void configure() throws Exception {
		config = Configuration.getGlobalConfiguration();
		image = IJ.getImage();
		if (filenamePriors == null)
			filenamePriors = FileUtil.myFileChoose(".xml", false);
		twoDPosReal = (ArrayList<ArrayList<double[]>>) XmlUtils.importFromXML(filenamePriors);
		this.fastRadialSymmetrySpace = null;
		this.allDetectedBeads = null;
		update3Dreference();
		update2Dreference();
		configured = true;
	}


	public void setParameters(double binThresh, double circularity, double lowGradThresh, double distance, String radii){

		String[] strings = radii.replaceAll("\\s*", "").replace("]", "").replace("[", "").split(",");
		ArrayList<Double> tmp = new ArrayList<Double>();
		for (int i = 0; i < strings.length; i++) {

			double h=0;
			try {
				h = Double.parseDouble(strings[i]);

			} catch (NumberFormatException e) {
				// TODO: handle exception
				e.printStackTrace();
				continue;
			}
			tmp.add(h);
		}

		if(tmp.size()<=0){
			tmp.add(3.0);
			System.out.println("No radii are given, using 3 as standard radius!");
		}

		if (distance != this.distance || 
				lowGradThresh != this.lowGradThresh || 
				this.circularity != circularity ||
				binThresh != this.binarizationThreshold )
		{
			this.distance = distance;
			this.lowGradThresh = lowGradThresh;
			this.circularity = circularity;
			this.binarizationThreshold = binThresh;
			configured = false;
		}

		boolean same = false;
		if (radiusOfBeads.length == tmp.size()){
			same = true;
			for (int i = 0; i < this.radiusOfBeads.length; i++) {
				if(radiusOfBeads[i]!=tmp.get(i)){
					same = false;
					break;
				}
			}
		}

		if (!same){
			radiusOfBeads = new double[tmp.size()];
			for (int i = 0; i < radiusOfBeads.length; i++) {
				radiusOfBeads[i]=tmp.get(i);
			}
			configured = false;
		}


	}

	public String getFilename(){
		return filenamePriors;
	}
	public void setFilename(String s){
		filenamePriors=s;
	}

	public void setImage(ImagePlus img){
		image = img;
	}

	public ArrayList<ArrayList<double[]>> getMeasuredTwoDPoints(){
		return twoDPosReal;
	}

	public ArrayList<ArrayList<double[]>> getReferenceTwoDPoints(){
		return twoDPosRef;
	}

	public ArrayList<double[]> getReferenceThreeDPoints(){
		return threeDPos;
	}

	public double[] getRadii(){
		return radiusOfBeads;
	}

	@Override
	public boolean isConfigured() {
		return configured;
	}
	
	public void setConfigured(boolean config) {
		configured = config;
	}



}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

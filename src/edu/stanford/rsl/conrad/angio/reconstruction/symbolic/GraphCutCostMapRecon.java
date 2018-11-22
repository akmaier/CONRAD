package edu.stanford.rsl.conrad.angio.reconstruction.symbolic;

import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.gui.PointRoi;
import ij.plugin.frame.RoiManager;

import java.util.ArrayList;
import java.util.Random;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.cuts.GraphCut;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.BranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonBranch;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.data.organization.AngiographicView;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;
//import edu.stanford.rsl.conrad.visualization.PointCloudViewer;
//import edu.stanford.rsl.conrad.visualization.ReprojectToRois;

public class GraphCutCostMapRecon {
	
	private int idx = 0;
		
	DataSet dataset = null;
	
	private ArrayList<AngiographicView> views = null;
	/** Initial view to start the epipolar matching. */
	private AngiographicView view1 = null;
	
	private double k1 = 30;	// mm used in World coordinates
	private double k2 = 32; // mm used in World Coordinates
	private double k3 = 9.6;// mm used in World Coordinates
	
	private double beta = 1; // regularization parameter for error function
	
	private int maxLabel = 512;
	private double labelCenterOffset = 0.55;
	private double sourceDetectorDistanceCoverage = 0.2;
	
	private int[] depthLabels = null;
	private ArrayList<Double> depth = null;
	private ArrayList<BranchPoint> points = new ArrayList<BranchPoint>();
	private ArrayList<int[]> neighbors = new ArrayList<int[]>();
	
	private boolean DEBUG = true;
		
	float currentFlow = Float.MAX_VALUE;
	
	//===============================================================
	//	Methods
	//===============================================================

	public static void main(String[] args){
		int caseID = 3;
		
		double refHeartPhase = 0.8;
				
		DataSets datasets = DataSets.getInstance();
		DataSet ds = datasets.getCase(caseID);
		
		String dir = ds.getDir()+"eval/" + String.valueOf(refHeartPhase)+"/";
		Projection[] ps = ProjMatIO.readProjMats(dir+"pMat.txt");
		Grid3D cm = ImageUtil.wrapImagePlus(IJ.openImage(dir+"distTrafo.tif"));
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(dir+"img.tif"));
		Angiogram prepAng = new Angiogram(img, ps, new double[ps.length]);
		prepAng.setReconCostMap(cm);
		ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
		Grid3D binImg = SkeletonUtil.costMapToVesselTreeImage(cm);
		for(int i = 0; i < ps.length; i++){
			skels.add(SkeletonUtil.binaryImgToSkel(binImg.getSubGrid(i), 0, false));
		}
		prepAng.setSkeletons(skels);
		
		int idx = 1;
		GraphCutCostMapRecon rec = new GraphCutCostMapRecon(ds, cm, idx, skels.get(idx), ps, new double[ps.length]);
		ArrayList<PointND> pts = rec.reconstruct();
		
		// TODO Wait for migration of visualization.ReprojectToRoise
		//		new PointCloudViewer("", pts);
		//		ReprojectToRois.reprojectPointList(ps, img, pts);
		
	}
	
	//===============================================================
	
	public GraphCutCostMapRecon(DataSet ds, Grid3D costMap, int refIdx, Skeleton skeleton, 
								Projection[] pMatrices, double[] primAngles){
		init(ds, costMap,skeleton,pMatrices,primAngles,refIdx);
	}
	
	
	/**
	 * Reconstructs the coronary artery tree using Alpha-Expansion moves and a soft-epipolar constraint.
	 * @return
	 */
	private ArrayList<PointND> reconstructAlphaExpansion(){
		
		System.out.println("Reconstructing.");
		float minMaxFlow = Float.MAX_VALUE;
		boolean cont = true;
		int iter = 0;
		while(cont && iter < 4){
			float flow = Float.MAX_VALUE;
			for(int alpha = maxLabel; alpha > 0; alpha--){
				flow = runAlphaExpansion(alpha);
				if(DEBUG){
					System.out.println("Loop "+String.valueOf(iter+1)+" - Flow at label "+alpha+" : "+flow);
				}
			}
			iter++;
			if(flow < minMaxFlow){
				minMaxFlow = flow;
				cont = true;
			}else{
				cont = false;
			}
		}
		// reconstruct the points using the geometry of the reference view
		ArrayList<PointND> reconPoints = new ArrayList<PointND>();
		for(int i = 0; i < points.size(); i++){
			reconPoints.add(view1.reconstruct(points.get(i), depthLabels[i], maxLabel));
		}
				
		return reconPoints;
	}
	
	
	

	/**
	 * Performs one Alpha-Expansion move. 
	 * @param alpha
	 * @return
	 */
	private float runAlphaExpansion(int alpha){
		long startTime = System.nanoTime();
		
		int numNodes = points.size();
		int numEdges = 0;
		ArrayList<Boolean> junc = new ArrayList<Boolean>();
		ArrayList<Boolean> equalLabel = new ArrayList<Boolean>();
		for(int i = 0; i < neighbors.size(); i++){
			if(Math.abs(depthLabels[neighbors.get(i)[0]] - depthLabels[neighbors.get(i)[1]]) > 1){
				numNodes += 1;
				numEdges += 2;
				equalLabel.add(false);
			}else{
				numEdges += 1;
				equalLabel.add(true);
			}
			boolean isJ = ( points.get(neighbors.get(i)[0]).isJUNCTION() || 
							points.get(neighbors.get(i)[1]).isJUNCTION()	);
			junc.add(isJ);
		}
		GraphCut gc = new GraphCut(numNodes, numEdges);
		
		// set the terminal weights of the graph
		for(int i = 0; i < points.size(); i++){
			gc.setTerminalWeights(	i, 
									calculateAlphaWeight(points.get(i), depthLabels[i], alpha),
									calculateNotAlphaWeight(points.get(i), depthLabels[i], alpha));
		}
		int auxTerminalIdx = points.size();
		for(int i = 0; i < neighbors.size(); i++){
			if(!equalLabel.get(i)){
				int[] idx = neighbors.get(i);
				gc.setTerminalWeights(	auxTerminalIdx, 
										Float.MAX_VALUE, 
										calculateSmoothnessTerm(depthLabels[idx[0]], depthLabels[idx[1]], junc.get(i)));
			}
		}
		
		// set the edge weights of the graph
		// edge weights are set for the auxiliary nodes
		int auxNodeIdx = points.size();
		for(int i = 0; i < neighbors.size(); i++){
			int[] idx = neighbors.get(i);
			boolean isJunc = junc.get(i);
			if(equalLabel.get(i)){
				// labels are equal, no auxiliary nodes involved
				gc.setEdgeWeight(	idx[0],
								 	idx[1],	
								 	calculateSmoothnessTerm(depthLabels[idx[0]], alpha, isJunc));
			}else{
				// calculate position of auxiliary node
				gc.setEdgeWeight(	auxNodeIdx,
									idx[0],	
									calculateSmoothnessTerm(depthLabels[idx[0]], alpha, isJunc));
				gc.setEdgeWeight(	auxNodeIdx,
									idx[1],	
									calculateSmoothnessTerm(depthLabels[idx[1]], alpha, isJunc));
				auxNodeIdx++;				
			}
				
		}
		// initiate max-flow / min-cut calculation
		float totalFlow = gc.computeMaximumFlow(false, null);
		// loop through nodes and check the terminal they are connected to
		// if connected to alpha, they keep their old label
		// if connected to not-alpha, they change their label to alpha
		int changedLabels = 0;
		if(totalFlow < currentFlow){
			currentFlow = totalFlow;
			for(int i = 0; i < points.size(); i++){
				if(gc.getTerminal(i) == GraphCut.Terminal.NOT_ALPHA){
					depthLabels[i] = alpha;
					changedLabels++;
				}
			}	
		}
		long endTime = System.nanoTime();
		if(DEBUG){
			float duration = (endTime - startTime) / 1000000f;
			System.out.println( "Calculation time for expansion step with terminal "+alpha+" : "
								+ duration+"ms = " + duration/1000/60 + "min");
			System.out.println("Number of labels changed to "+ alpha +" : "+changedLabels);
		}
		return totalFlow;
	}
	
	/**
	 * Calculates the regularizing term V, that punishes different labels at neighboring points if they are not junctions.
	 * @param labelP
	 * @param labelQ
	 * @param isJunction
	 * @return
	 */
	private float calculateSmoothnessTerm(int labelP, int labelQ, boolean isJunction){
		double labP = labelP / (maxLabel-1.0d) * view1.getSourceToDetectorDistance();
		double labQ = labelQ / (maxLabel-1.0d) * view1.getSourceToDetectorDistance();
		
		double e = Math.abs(labP-labQ);
		if(!isJunction){
			return (float)(beta*Math.min(e, k2));
		}else{
			return (float)(beta*Math.min(e, k3));
		}
	}
	
	/**
	 * Calculates the terminal weight to the alpha terminal.
	 * @param point
	 * @param label
	 * @param alpha
	 * @return
	 */
	private float calculateAlphaWeight(BranchPoint point, int label, int alpha){
		return calculateImageTerm(point, alpha);
	}
	
	/**
	 * Calculates the terminal weight to the not-alpha terminal.
	 * @param point
	 * @param label
	 * @param alpha
	 * @return
	 */
	private float calculateNotAlphaWeight(BranchPoint point, int label, int alpha){
		if(label == alpha){
			return Float.MAX_VALUE;
		}else{
			return calculateImageTerm(point, label);
		}
	}
	
	/**
	 * Calculates the image / point similarity measure.
	 * @param point
	 * @param label
	 * @return
	 */
	private float calculateImageTerm(BranchPoint point, int label){
		PointND point3D = view1.reconstruct(point, label, maxLabel);
		double projErr = 0;
		for(int i = 0; i < views.size(); i++){
			PointND projected = views.get(i).project(point3D);
			double err = interpolateLinear(views.get(i).getProjection(), projected.get(0), projected.get(1));
			projErr += Math.min(err, k1);
		}
		projErr /= (views.size());
		return (float)projErr;
	}
	
	private double interpolateLinear(Grid2D grid, double x, double y) {
		if (x < 0 || x > grid.getSize()[0]-1 || y < 0 || y > grid.getSize()[1]-1)
			return k1;
		
		int lower = (int) Math.floor(y);
		double d = y - lower; // d is in [0, 1)

		float val =  (float) (
				(1.0-d)*InterpolationOperators.interpolateLinear(grid.getSubGrid(lower), x)
				+ ((d != 0.0) ? d*InterpolationOperators.interpolateLinear(grid.getSubGrid(lower+1), x) : 0.0)
		);
		return val;
	}

	
	public ArrayList<PointND> reconstructNaive(){
		ArrayList<PointND> recon = new ArrayList<PointND>();
		for(int i = 0; i < points.size(); i++){
			double minPointError = Double.MAX_VALUE;
			int minErrLabel = 0;
			for(int j = 0; j < maxLabel; j++){
				double errAtLabel = calculateImageTerm(points.get(i), j);
				if(errAtLabel < minPointError){
					minPointError = errAtLabel;
					minErrLabel = j;
				}
			}
			recon.add(view1.reconstruct(points.get(i), minErrLabel, maxLabel));
		}
		return recon;			
	}
	
	/**
	 * Reconstruct the 3D coronary artery tree.
	 * @return
	 */
	public ArrayList<PointND> reconstruct(){
		ArrayList<PointND> reconPoints = null;
		
		reconPoints = reconstructAlphaExpansion();
		
		this.depth = new ArrayList<Double>();
		for(int i = 0; i < reconPoints.size(); i++){
			depth.add(view1.getDepthFromLabel(points.get(i).get2DPointND(), depthLabels[i]));
		}
		
		return reconPoints;
	}
	
	
	
	private int[] initVariables( Skeleton skel1,
								ArrayList<BranchPoint> points,
								ArrayList<int[]> neighbors){
		
		System.out.println("Calculating neighbors for all branches.");
		int linIdx = 0;
		for(int i = 0; i < skel1.size(); i++){
			SkeletonBranch branch = skel1.get(i);
			int nBranch = branch.size();
			for(int j = 0; j < nBranch; j++){
				points.add(branch.get(j));
				ArrayList<Integer> neigh = getNeighbors(branch.get(j), skel1, true);
				for(int k = 0; k < neigh.size(); k++){
					if(neigh.get(k) > linIdx){
						neighbors.add(new int[]{linIdx, neigh.get(k)});
					}
				}
				linIdx++;
			}
		}
		int[] depthLabels = new int[points.size()];
		Random rand = new Random();
		for(int i = 0; i < points.size(); i++){
			depthLabels[i] = rand.nextInt(maxLabel);
		}
		return depthLabels;
	}
	
	/**
	 * Determines neighboring points of p in the skeleton. It is possible to determine neighbors using 
	 * four-connectedness and eight-connectedness by providing the appropriate boolean flag.
	 * The method returns a List of the linearized indices of the neighboring points.
	 * @param p
	 * @param skel
	 * @param eightConnected
	 * @return
	 */
	private ArrayList<Integer> getNeighbors(BranchPoint p, Skeleton skel, boolean eightConnected){
		ArrayList<Integer> neighbors4 = new ArrayList<Integer>();
		ArrayList<Integer> neighbors8 = new ArrayList<Integer>();
		
		int linearIndex = 0;
		
		for(int i = 0; i < skel.size(); i++){
			SkeletonBranch branch = skel.get(i);
			int nBranch = branch.size();
			for(int j = 0; j < nBranch; j++){
				BranchPoint nc = branch.get(j);
				if((int)Math.abs(p.x-nc.x) == 1){
					if((int)Math.abs(p.y-nc.y) == 0){
						neighbors4.add(linearIndex);
						neighbors8.add(linearIndex);
					}
					if((int)Math.abs(p.y-nc.y) == 1){
						neighbors8.add(linearIndex);
					}
				}else if((int)Math.abs(p.y-nc.y) == 1){
					if((int)Math.abs(p.x-nc.x) == 0){
						neighbors4.add(linearIndex);
						neighbors8.add(linearIndex);
					}
					if((int)Math.abs(p.x-nc.x) == 1){
						neighbors8.add(linearIndex);
					}
				}
				linearIndex++;
			}
		}	
		return (eightConnected)?neighbors8:neighbors4;
	}
	/**
	 * Initializes class members
	 * @param cm
	 * @param skeleton
	 * @param pMatrices
	 * @param primAngles
	 * @param idx
	 */
	private void init(DataSet ds, Grid3D cm, Skeleton skeleton,
					  Projection[] pMatrices, double[] primAngles, int idx){
		this.dataset = ds;
		this.maxLabel = ds.getRecoSet().getNumDepthLabels();
		this.labelCenterOffset = ds.getRecoSet().getLabelCenterOffset();
		this.sourceDetectorDistanceCoverage = ds.getRecoSet().getSourceDetectorDistanceCoverage();
		
		this.idx = idx;
		Grid2D v1g = cm.getSubGrid(idx);
		v1g.setSpacing(cm.getSpacing()[0],cm.getSpacing()[1]);
		this.view1 = new AngiographicView(pMatrices[idx], v1g, skeleton, primAngles[idx]);
		view1.setupKdTree();
		view1.setDepthLabelCenterOffset(labelCenterOffset);
		view1.setDepthLabelRange(sourceDetectorDistanceCoverage);
		// set the rest of the views
		this.views = new ArrayList<AngiographicView>();
		
		for(int i = 0; i < pMatrices.length; i++){
			if( (i == idx)){
				continue;
			}else{
				Grid2D vIg = cm.getSubGrid(i);
				vIg.setSpacing(cm.getSpacing()[0],cm.getSpacing()[1]);
				AngiographicView view = new AngiographicView(pMatrices[i], vIg, null, primAngles[i]);
				view.setDepthLabelCenterOffset(labelCenterOffset);
				view.setDepthLabelRange(sourceDetectorDistanceCoverage);				
				views.add(view);
			}
		}
		
		// write skeleton in list format, search neighbors for constraints and initialize depth labels
		this.depthLabels = initVariables(view1.getSkeleton(),points,neighbors);
				
	}
	
	public int getDimension(){
		return this.depthLabels.length;
	}
	
	public void setDebug(boolean deb){
		this.DEBUG = deb;
	}
	
	public double[] getUpperBound(){
		double[] upper = new double[this.depthLabels.length];
		for(int i = 0; i < depthLabels.length; i++){
			upper[i] = 1;
		}
		return upper;
	}
	
	public int[] getDepthLabels(){
		return this.depthLabels;
	}
	
	public ArrayList<BranchPoint> getBranchPoints(){
		return points;
	}
	
	public ArrayList<Double> getDepth(){
		return depth;
	}
	
	public ArrayList<AngiographicView> getAngiographicViews(){
		ArrayList<AngiographicView> v = new ArrayList<AngiographicView>();
		int count = 0;
		for(int i = 0; i < views.size()+1; i++){
			if(i == this.idx){
				v.add(view1);
			}else{
				v.add(views.get(count));
				count++;
			}
		}
		return v;
	}
		
	public void displayProjections(ArrayList<PointND> pts){
		Grid3D proj = new Grid3D(view1.getProjection().getSize()[0],view1.getProjection().getSize()[1], views.size());
		for(int v = 0; v < views.size(); v++){
			proj.setSubGrid(v, views.get(v).getProjection());
		}
		ImagePlus imp = ImageUtil.wrapGrid3D(proj,"");
		RoiManager manager = new RoiManager();
		int count = 0; 
		for(int v = 0; v < views.size(); v++){
			for(int i = 0; i < pts.size(); i++){
				PointND p = views.get(v).project(pts.get(i));
				PointRoi pRoi = new PointRoi(p.get(0),p.get(1));
				pRoi.setPosition(v+1);
				manager.add(imp, pRoi, count);
				count++;
			}
		}
		imp.getProcessor().setMinAndMax(0, 10);
		imp.show();
		Prefs.showAllSliceOnly = true;
		manager.runCommand("Show All");
	}
	
	
}

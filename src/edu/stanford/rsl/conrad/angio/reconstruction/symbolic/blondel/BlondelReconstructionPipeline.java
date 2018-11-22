package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel;

import ij.IJ;
import ij.ImageJ;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.MinimumSpanningTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.PreprocessingPipeline;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.io.EdgeListIO;
import edu.stanford.rsl.conrad.angio.util.io.PointAndRadiusIO;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;

public class BlondelReconstructionPipeline {

	// Epipolar geometry parameters
	private static double maxDistance = 1.5;
	// this one might need optimization
	// weighting parameter for the optimization
	private static double alpha = 2.5;
	// thresholds for the internal energy term
	private static int dh = 50;
	private static int dl = 2;
	
	DataSet ds;
	double refHeartPhase = 0;
	String outputDir;
	boolean writeOutput = false;
	
	boolean writeIntermediate = false;
	String intermediateFile = "";
	
	public void setWriteIntermediate(boolean b, String file){
		this.writeIntermediate = b;
		this.intermediateFile = file;
	}
	
	public static void main(String[] args) throws IOException {
		
		int caseID = 50;
		
		double refHeartPhase = 0.9;
		
		
		DataSets datasets = DataSets.getInstance();
		DataSet ds = datasets.getCase(caseID);
		
//		BlondelReconstructionPipeline recoPipe = 
//			new BlondelReconstructionPipeline(ds, refHeartPhase, dir, true);
//		recoPipe.useExhaustiveExtension(true);
//		ArrayList<ArrayList<Edge>> edges = recoPipe.run();
		
		String testdir = ".../";
		String dir = testdir + String.valueOf(refHeartPhase)+"/";
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
		
		BlondelReconstructionPipeline recoPipe = 
				new BlondelReconstructionPipeline(ds, refHeartPhase, null, false);
		ArrayList<ArrayList<Edge>> edges = recoPipe.run(prepAng);		
		
		// TODO Wait for migration of edge viewer;
//		EdgeViewer.renderEdgesComponents(edges);
	}
	
	public BlondelReconstructionPipeline(DataSet ds, double heartPhase, String outDir, boolean write){
		this.ds = ds;
		this.refHeartPhase = heartPhase;
		this.outputDir = outDir;
		this.writeOutput = write;
	}
	
	public ArrayList<ArrayList<Edge>> run(Angiogram prepAng){
		return runInternal(prepAng);
	}
	
	public ArrayList<ArrayList<Edge>> run(){
		Angiogram prepAng;
		String dir = this.outputDir;
		if(refHeartPhase<0){
			dir += "all/";
		}else{
			dir += String.valueOf(refHeartPhase)+"/";
		}
		File fTest = new File(dir);
		if(!fTest.exists()){
			PreprocessingPipeline prepPipe = new PreprocessingPipeline(ds);
			prepPipe.setRefHeartPhase(refHeartPhase);
			prepPipe.evaluate();
			prepAng = prepPipe.getPreprocessedAngiogram();
		}else{
			Projection[] ps = ProjMatIO.readProjMats(dir+"pMat.txt");
			Grid3D cm = ImageUtil.wrapImagePlus(IJ.openImage(dir+"distTrafo.tif"));
			Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(dir+"img.tif"));
			prepAng = new Angiogram(img, ps, new double[ps.length]);
			prepAng.setReconCostMap(cm);
			ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
			Grid3D binImg = SkeletonUtil.costMapToVesselTreeImage(cm);
			for(int i = 0; i < ps.length; i++){
				skels.add(SkeletonUtil.binaryImgToSkel(binImg.getSubGrid(i), 0, false));
			}
			prepAng.setSkeletons(skels);
		}
		return runInternal(prepAng);
	}
	
	
	public ArrayList<ArrayList<Edge>> runInternal(Angiogram prepAng){		
		long startTime = System.nanoTime();
		
		EpipolarReconGeometry geometryBetweenAllViews = new EpipolarReconGeometry(prepAng.getPMatrices(),
				prepAng.getSkeletons(),false,true);
		geometryBetweenAllViews.setCostMap(prepAng.getReconCostMap());
		geometryBetweenAllViews.setExhaustiveReconParameters(
				ds.getRecoSet().getMaxReprojectionError(), ds.getRecoSet().getSuppressionRadius());
		geometryBetweenAllViews.setParameters(maxDistance, alpha, dl, dh);
		geometryBetweenAllViews.evaluate();
		// get the optimal correspondences as a arraylist of 3D points
		ArrayList<PointND> reconPts = geometryBetweenAllViews.getCorrAfterDijkst();
		
		ArrayList<ArrayList<Edge>> edges;
		
		MinimumSpanningTree mst = new MinimumSpanningTree(reconPts);
		mst.run();
		edges =  mst.getMstHierarchical();
		
		long time = System.nanoTime() - startTime;
		time = TimeUnit.NANOSECONDS.toMillis(time);
		System.out.println("Reconstruction time was: " + (float)time/1000.0 +"s.");
		
		if(writeOutput){
			String outputDir = this.outputDir + String.valueOf(refHeartPhase)+"/";
			File f = new File(outputDir);
			if(!f.exists()){
				f.mkdirs();
			}
			PointAndRadiusIO prio = new PointAndRadiusIO();
			ArrayList<Double> radRef = new ArrayList<Double>();
			for(int i = 0; i < reconPts.size(); i++){
				radRef.add(1.0d);
			}
			IJ.saveAsTiff(prepAng.getProjectionsAsImP(), outputDir+"img.tif");
			IJ.saveAsTiff(ImageUtil.wrapGrid3D(prepAng.getReconCostMap(),""), outputDir+"distTrafo.tif");
			prio.write(outputDir+"pts.txt", reconPts, radRef);
			ProjMatIO.writeProjTable(prepAng.getPMatrices(), outputDir+"pMat.txt");
			EdgeListIO.write(outputDir+"edges.txt", edges);
		}
		//EdgeViewer.renderEdgesComponents(edges);
		return edges;
	}
	
	public static ArrayList<ArrayList<Edge>> toEdgeList(ArrayList<PointND> correspondences){
		ArrayList<ArrayList<Edge>> edge = new ArrayList<ArrayList<Edge>>();
		ArrayList<Edge> l =	new ArrayList<Edge>();
		for (int i = 0; i < correspondences.size()-1; i++) {
			Edge e = new Edge(correspondences.get(i), correspondences.get(i+1));
			if(e.getLength() < 5){
				l.add(e);
			}
		}
		edge.add(l);
		return edge;
	}
	
}

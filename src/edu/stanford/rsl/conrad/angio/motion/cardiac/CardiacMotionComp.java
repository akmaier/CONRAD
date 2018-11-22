/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.motion.cardiac;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.gui.PointRoi;
import ij.plugin.frame.RoiManager;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.io.EcgIO;
import edu.stanford.rsl.conrad.angio.util.io.EdgeListIO;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;

public class CardiacMotionComp extends GraMoOptimFunc{

	private ArrayList<Translation> allowedTranslations;
	private int[] numShifts;
	private double[] magShift; // in mm
	
	// Paramters for Bspline Motion Model
	private int degree = 2;
	// note that in this case, the first and last control point will not be optimized
	private int numControlPoints = 5;
	private double[] uKnots;
	private ArrayList<BSpline> splines;
	
	private double neighborhood = 2.5; // in mm
	private ArrayList<int[]> neighbors;
	private ArrayList<Double> neighborWeights;
	
	private boolean useMeanError = true;
	private double k1 = 2.5; // in mm
	private int subSamplingFactor = 1;
	private double[] splineSamplingLocations;
	private int[] samplingIndices;
	private ImagePlus reducedGrid;
	
	private Grid3D img;
	private Grid3D costMap;
	private Projection[] pMats;
	private double[] ecg;
	private ArrayList<ArrayList<Edge>> edges;
	private double refHp;
	
	private ImageJ ij = null;
	private RoiManager manage = null;
	
	public static void main(String[] args) {
		String dir = ".../";
		
		Angiogram ang = CardiacMotionComp.getSavedAngio(dir);
		ArrayList<ArrayList<Edge>> edges = EdgeListIO.read(dir+"test_moco/edges.txt");
		//ArrayList<ArrayList<Edge>> edges = EdgeListIO.read(dir+"test_moco/test.txt");
		
		
		int[] samples = new int[]{7, 7, 7};
		double[] range = new double[]{3.5, 3.5, 3.5};
		
		CardiacMotionComp moco = new CardiacMotionComp(ang, edges, 0.1, samples, range);
		
		GraMoCa optim = new GraMoCa(moco);
		optim.run();
		
	}
	
	public CardiacMotionComp(Angiogram ang, ArrayList<ArrayList<Edge>> edges, double hp, int[] samples, double[] range) {
		this.img = ang.getProjections();
		this.costMap = ang.getReconCostMap();
		this.pMats = ang.getPMatrices();
		this.ecg = ang.getEcg();
		this.edges = edges;
		this.refHp = hp;
		numShifts = samples;
		magShift = range;
		
	}
	
	@Override
	public void init(){
		// prepare resampling and make sure that we normalize w.r.t. the heart phase
		// the reconstruction has been performed. This is necessary, because we will
		// currently not optimize for shifts at that phase and use clamped splines
		int numSamples = (int)Math.ceil(pMats.length/((double)subSamplingFactor));
		splineSamplingLocations = new double[numSamples];
		samplingIndices = new int[numSamples];
		Grid3D reduced = new Grid3D(img.getSize()[0], img.getSize()[1], numSamples);
		reduced.setSpacing(img.getSpacing());
		int count = 0;
		for(int i = 0; i < pMats.length; i+=subSamplingFactor){
			samplingIndices[count] = i;
			double val = ecg[count] - refHp;
			splineSamplingLocations[count] = (val<0)?(1+val):val;
			reduced.setSubGrid(count, img.getSubGrid(i));
			count++;
		}
		reducedGrid = ImageUtil.wrapGrid3D(reduced, "Reduced Projection Stack");
		// setup control points
		splines = new ArrayList<BSpline>();
		double[] uVec = new double[numControlPoints];
		for(int i = 0; i < numControlPoints; i++){
			uVec[i] = i/(numControlPoints-1.0);
		}
		ArrayList<PointND> pts = pointsFromEdgeLists(edges);
		for(int i = 0; i < pts.size(); i++){
			ArrayList<PointND> cntrl = new ArrayList<PointND>();
			for(int j = 0; j < numControlPoints; j++){
				cntrl.add(pts.get(i).clone());
			}
			BSpline spline = splineFromControlPoints(cntrl);
			splines.add(spline);
		}
		// initialize shifts
		allowedTranslations = new ArrayList<Translation>();
		for(int k = 0; k < numShifts[2]; k++){
			double shiftsZ = (numShifts[2]==1)?0.0f:(magShift[2]*(2f/(numShifts[2]-1)*k-1));
			for(int j = 0; j < numShifts[1]; j++){
				double shiftsY = (numShifts[1]==1)?0.0f:(magShift[1]*(2f/(numShifts[1]-1)*j-1));
				for(int i = 0; i < numShifts[0]; i++){
					double shiftsX = (numShifts[0]==1)?0.0f:(magShift[0]*(2f/(numShifts[0]-1)*i-1));
					allowedTranslations.add(new Translation(shiftsX, shiftsY, shiftsZ));		
				}
			}
		}
		int zeroLabel = 0;
		double minShift = Double.MAX_VALUE;
		for(int i = 0; i < allowedTranslations.size(); i++){
			double val = allowedTranslations.get(i).getData().normL2();
			if(val < minShift){
				minShift = val;
				zeroLabel = i;
			}
		}
	
		// initialize neighbors
		initializeNeighborhoods(pts);
		// determine number of parameters:
		// Note: clamped spline and we have a reconstruction that is fixed at that phase!
		int numParams = splines.size() * (numControlPoints - 2);
		labels = new int[numParams];
		for(int i = 0; i < labels.length; i++){
			labels[i] = zeroLabel;
		}
	}
	

	private void initializeNeighborhoods(ArrayList<PointND> pts) {
		this.neighbors = new ArrayList<int[]>();
		this.neighborWeights = new ArrayList<Double>();
		for (int i = 0; i < pts.size(); i++) {
			PointND p1 = pts.get(i);
			for(int j = 0; j < i; j++){
				PointND p2 = pts.get(j);
				double dist = p1.euclideanDistance(p2);
				if(dist < neighborhood){
					for(int k = 1; k < numControlPoints-1; k++){
						int idx1 = i*(numControlPoints-2) + k -1;
						int idx2 = j*(numControlPoints-2) + k -1;
						neighbors.add(new int[]{idx1, idx2});
						neighborWeights.add(1.0);
//						if(k < numControlPoints-2){
//							neighbors.add(new int[]{idx1, idx1+1});
//							neighborWeights.add(1.0);
//						}
					}
				}
			}
		}
	}

	@Override
	public int[] getLabels() {
		return labels;
	}

	@Override
	public int getNumLabels() {
		return allowedTranslations.size();
	}

	@Override
	public ArrayList<int[]> getNeighbors() {
		return neighbors;
	}

	@Override
	public ArrayList<Double> getNeighborWeights() {
		return neighborWeights;
	}

	@Override
	public double computeDataTerm(int idx, int labelP) {
		int splineIdx = idx/(numControlPoints-2);
		int controlPointIdx = idx - splineIdx*(numControlPoints-2);
		// adapt for not considering first and last
		controlPointIdx += 1;
		BSpline sp = new BSpline(splines.get(splineIdx));
		ArrayList<PointND> cpts = sp.getControlPoints();
		cpts.set(controlPointIdx, allowedTranslations.get(labelP).transform(cpts.get(controlPointIdx)));
		sp = splineFromControlPoints(cpts);
		double err = 0;
		double maxErr = -Double.MAX_VALUE;
		ArrayList<PointND> pts = samplePointsFromSpline(sp);
		for(int i = 0; i < samplingIndices.length; i++){
			SimpleVector p = new SimpleVector(2);
			pMats[samplingIndices[i]].project(pts.get(i).getAbstractVector(), p);
			double val = InterpolationOperators.interpolateLinear(
					costMap.getSubGrid(samplingIndices[i]), p.getElement(0),p.getElement(1));
			err += Math.min(val, k1);	
			maxErr = Math.max(val, maxErr);
		}
		if(useMeanError){
			return err/samplingIndices.length;//
		}else{
			return maxErr;//
		}
	}

	@Override
	public double computeNeighborhoodTerm(int labelP, int labelQ) {
		if(labelP == labelQ){
			return 0.0;
		}else{
			SimpleVector s = allowedTranslations.get(labelP).getData().clone();
			s.subtract(allowedTranslations.get(labelQ).getData());
			return s.normL2();
		}
	}

	@Override
	public void updateLabels(int[] labels){
		this.labels = labels;
	}
	
	@Override
	public void updateMotionState(){
		for(int i = 0; i < splines.size(); i++){
			BSpline sp = new BSpline(splines.get(i));
			ArrayList<PointND> cpts = sp.getControlPoints();
			for(int k = 1; k < numControlPoints-1; k++){
				int labelIdx = i*(numControlPoints-2)+k;
				// correct for offset
				labelIdx -= 1;
				cpts.set(k, allowedTranslations.get(labels[labelIdx]).transform(cpts.get(k)));
			}
			sp = splineFromControlPoints(cpts);
			splines.set(i, sp);
		}
	}
	
	@Override
	public void updateVisualization() {
		if(this.ij == null){
			ij = new ImageJ();
			reducedGrid.show();
		}
		if(manage == null){
			manage = new RoiManager();
			Prefs.showAllSliceOnly = true;
			
		}else{
			manage.runCommand(reducedGrid, "Select All");
			manage.runCommand(reducedGrid, "Delete");
		}
		ArrayList<BSpline> updatedSplines = new ArrayList<BSpline>();
		for(int j = 0; j < splines.size(); j++){
			BSpline sp = new BSpline(splines.get(j));
			ArrayList<PointND> cpts = sp.getControlPoints();
			for(int k = 1; k < numControlPoints-1; k++){
				int labelIdx = j*(numControlPoints-2)+k;
				// correct for offset
				labelIdx -= 1;
				cpts.set(k, 
						allowedTranslations.get(labels[labelIdx]).transform(cpts.get(k)));
			}
			sp = splineFromControlPoints(cpts);
			updatedSplines.add(sp);
		}
		int count = 0;
		for(int i = 0; i < samplingIndices.length; i++){
			for(int j = 0; j < splines.size(); j++){
				PointND p3D = updatedSplines.get(j).evaluate(splineSamplingLocations[i]);
				SimpleVector p = new SimpleVector(2);
				pMats[samplingIndices[i]].project(p3D.getAbstractVector(), p);
				PointRoi pRoi = new PointRoi(p.getElement(0),p.getElement(1));
				pRoi.setPosition(i+1);
				manage.add(reducedGrid, pRoi, count);
				count++;
			}
		}
		manage.runCommand("Show All");
	}
	
	public static Angiogram getSavedAngio(String dir){
		Projection[] ps = ProjMatIO.readProjMats(dir+"pMat.txt");
		Grid3D cm = ImageUtil.wrapImagePlus(IJ.openImage(dir+"distTrafo.tif"));
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(dir+"img.tif"));
		double[] ecg = EcgIO.readEcg(dir+"cardLin.txt");
		Angiogram prepAng = new Angiogram(img, ps, ecg);
		prepAng.setReconCostMap(cm);
		ArrayList<Skeleton> skels = new ArrayList<Skeleton>();
		Grid3D binImg = SkeletonUtil.costMapToVesselTreeImage(cm);
		for(int i = 0; i < ps.length; i++){
			skels.add(SkeletonUtil.binaryImgToSkel(binImg.getSubGrid(i), 0, false));
		}
		prepAng.setSkeletons(skels);
		return prepAng;
	}
	
	private ArrayList<PointND> samplePointsFromSplines(double u) {
		ArrayList<PointND> pts = new ArrayList<PointND>();
		for (int i = 0; i < splines.size(); i++) {
			pts.add(splines.get(i).evaluate(u));
		}
		return pts;
	}
	
	private ArrayList<PointND> samplePointsFromSpline(BSpline sp) {
		ArrayList<PointND> pts = new ArrayList<PointND>();
		for (int i = 0; i < splineSamplingLocations.length; i++) {
			pts.add(sp.evaluate(splineSamplingLocations[i]));
		}
		return pts;
	}
	
	private BSpline splineFromControlPoints(ArrayList<PointND> cntrl){
		if (uKnots == null){
			int n = cntrl.size();
			double[] parameters = new double[n];
			for (int i = 0; i < n; i++){
				parameters[i] = i / (n - 1.0);
			}		
			int k = n + degree + 1; //number of knots
			uKnots = new double[k];
			// compute the knot vector
			for (int i = 0; i <= degree; i++) {
				uKnots[i] = 0;
				uKnots[k-i-1] = 1;
			}		
			for (int j = 1; j <= n - degree - 1; j++) {
				double sum = 0;
				for (int i = j; i < j + degree; i++){
					sum += parameters[i];
				}
				uKnots[j + degree] =  sum / degree;
			}
		}
		BSpline bsp = new BSpline(cntrl, uKnots);		
		return bsp;
	}
	
	private ArrayList<PointND> pointsFromEdgeLists(ArrayList<ArrayList<Edge>> edges) {
		ArrayList<PointND> pts = new ArrayList<PointND>();
		for (int i = 0; i < edges.size(); i++) {
			for(int j = 0; j < edges.get(i).size(); j++){
				if(j == 0){
					pts.add(edges.get(i).get(j).getPoint());
				}
				pts.add(edges.get(i).get(j).getEnd());
			}
		}
		return pts;
	}
}

/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.organization;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.BranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonBranch;

public class AngiographicView {
	
	private Projection pMat = null;
	private SimpleVector cameraCenter = null;
	private Grid2D projection = null;
	private int sizeU;
	private int sizeV;
	private double spaceU = 1;
	private double spaceV = 1;
	private Skeleton skel = null;
	private double primAngle = 0;
	
	private double sourceDetectorDist = 0;
	
	private double depthLabelRange = 1;
	private double depthLabelCenterOffset = 0;
	
	
	private KDTree2D<ComparablePoint2D> kdTree = null;
	
	public AngiographicView(Projection m, Grid2D img, Skeleton s, double angle){
		this.pMat = m;
		this.cameraCenter = pMat.computeCameraCenter();
		this.projection = img;
		this.sizeU = img.getWidth();
		this.sizeV = img.getHeight();
		this.spaceU = img.getSpacing()[0];
		this.spaceV = img.getSpacing()[1];
		this.skel = s;
		this.primAngle = angle;
		this.sourceDetectorDist = pMat.computeSourceToDetectorDistance(new SimpleVector(spaceU, spaceV))[0];
	}
	
	public void setupKdTree(){
		ArrayList<ComparablePoint2D> compPoints = new ArrayList<ComparablePoint2D>();
		for(int i = 0; i < skel.size(); i++){
			SkeletonBranch branch = skel.get(i);
			int nBranch = branch.size();
			for(int j = 0; j < nBranch; j++){
				compPoints.add(new ComparablePoint2D(branch.get(j).get2DPointND()));
			}
		}
		kdTree = new KDTree2D<ComparablePoint2D>(compPoints); 
	}
	
	public PointND project(PointND point3D){
		SimpleMatrix mat = pMat.computeP();
		PointND p3Dhom = toHomogeneous(point3D);
		SimpleVector projected = SimpleOperators.multiply(mat, p3Dhom.getAbstractVector());
		return toCanonical(new PointND(projected.copyAsDoubleArray()));
	}
	
	public double projectPoint(PointND point3D){
		SimpleMatrix mat = pMat.computeP();
		PointND p3Dhom = toHomogeneous(point3D);
		SimpleVector projected = SimpleOperators.multiply(mat, p3Dhom.getAbstractVector());
		PointND idx =  toCanonical(new PointND(projected.copyAsDoubleArray()));
		return InterpolationOperators.interpolateLinear(projection, idx.get(0), idx.get(1));
	}
	
	public ArrayList<PointND> projectAll(ArrayList<PointND> points){
		ArrayList<PointND> projected = new ArrayList<PointND>();
		for(int i = 0; i < points.size(); i++){
			PointND projP = project(points.get(i));
			projected.add(projP);
		}
		return projected;
	}
	
	public Grid2D project(ArrayList<PointND> points){
		Grid2D projected = new Grid2D(this.projection.getWidth(), this.projection.getHeight());
		for(int i = 0; i < points.size(); i++){
			PointND projP = project(points.get(i));
			projected.setAtIndex((int)projP.get(0), (int)projP.get(1), 1);
		}
		return projected;
	}
	
	public double calculateScaleFactor(PointND p3D, PointND p2D){
		
		PointND detectorPoint3D = pMat.computeDetectorPoint(
				cameraCenter, p2D.getAbstractVector(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dir = detectorPoint3D.getAbstractVector().clone();
		dir.subtract(cameraCenter);
				
		SimpleVector p = new SimpleVector(p3D.getCoordinates());
		p.subtract(cameraCenter);
		double scale = p.normL2() / dir.normL2() * spaceU;
		return scale;
	}
	
	/**
	 * Reconstructs a 3D point from a detector coordinate and a depth label.
	 * The orientation is calculated from the world coordinate of the detector pixel position and the camera center, 
	 * the 3D point is then calculated as from the vectorial line equation using the depth label as parameter.
	 * While depth labels < 0 or > 1 are possible in principle, they do not necessarily make sense.  
	 * @param detectorCoord
	 * @param depthLabel
	 * @return
	 */
	public PointND reconstruct(PointND detectorCoord, double depthLabel){
		PointND detectorPoint3D = pMat.computeDetectorPoint(
				cameraCenter, detectorCoord.getAbstractVector(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dir = detectorPoint3D.getAbstractVector().clone();
		dir.subtract(cameraCenter);
		dir.normalizeL2();
		
		PointND principalPoint = pMat.computeDetectorPoint(
				cameraCenter, pMat.getPrincipalPoint(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dirPP = principalPoint.getAbstractVector().clone();
		dirPP.subtract(cameraCenter);
		dirPP.normalizeL2();
		
		double cosangle = SimpleOperators.multiplyInnerProd(dir, dirPP);
		
		double factor = depthLabelCenterOffset*sourceDetectorDist
					  	+ sourceDetectorDist*depthLabelRange*(depthLabel - 1/2);
		factor /= cosangle;
		
		SimpleVector point = cameraCenter.clone();
		point.add(dir.multipliedBy(factor));
		return new PointND(point.copyAsDoubleArray());
	}
	
	/**
	 * Reconstructs a 3D point from a detector coordinate and an absolute depth.
	 * The orientation is calculated from the world coordinate of the detector pixel position and the camera center, 
	 * the 3D point is then calculated as from the vectorial line equation using the depth.  
	 * @param detectorCoord
	 * @param depthLabel
	 * @return
	 */
	public PointND reconstructAbsoluteDepth(PointND detectorCoord, double depth){
		PointND detectorPoint3D = pMat.computeDetectorPoint(
				cameraCenter, detectorCoord.getAbstractVector(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dir = detectorPoint3D.getAbstractVector().clone();
		
		dir.normalizeL2();
		SimpleVector point = cameraCenter.clone();
		point.add(dir.multipliedBy(depth));
		return new PointND(point.copyAsDoubleArray());
	}
	
	public double getDepth(PointND p3D){
		PointND p2D = this.project(p3D);
		PointND detectorPoint3D = pMat.computeDetectorPoint(
				cameraCenter, p2D.getAbstractVector(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dir = detectorPoint3D.getAbstractVector().clone();
		dir.subtract(cameraCenter);
				
		SimpleVector p = new SimpleVector(p3D.getCoordinates());
		p.subtract(cameraCenter);
		return p.normL2();
		
	}
	
	public double getDepthFromLabel(PointND p2D, int depthLabel){
		PointND detectorPoint3D = pMat.computeDetectorPoint(
				cameraCenter, p2D.getAbstractVector(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dir = detectorPoint3D.getAbstractVector().clone();
		dir.subtract(cameraCenter);
		PointND principalPoint = pMat.computeDetectorPoint(
				cameraCenter, pMat.getPrincipalPoint(), sourceDetectorDist, spaceU, spaceV, sizeU, sizeV);
		SimpleVector dirPP = principalPoint.getAbstractVector().clone();
		dirPP.subtract(cameraCenter);
		dirPP.normalizeL2();
		
		double cosangle = SimpleOperators.multiplyInnerProd(dir, dirPP);
		
		double factor = depthLabelCenterOffset*sourceDetectorDist
					  	+ sourceDetectorDist*depthLabelRange*(depthLabel - 1/2);
		factor *= cosangle;
		return factor;		
	}
	
	public PointND reconstruct(PointND detectorCoord, int depthLabel, int maxLabel){
		double label = (double)depthLabel / (maxLabel-1.0d);
		return reconstruct(detectorCoord,label);
	}
	
	public PointND reconstruct(BranchPoint detectorCoord, int depthLabel, int maxLabel){
		double label = (double)depthLabel / (maxLabel-1.0d);
		return reconstruct(detectorCoord,label);
	}
	
	/**
	 * Reconstructs a 3D point from a detector coordinate and a depth label.
	 * The orientation is calculated from the world coordinate of the detector pixel position and the camera center, 
	 * the 3D point is then calculated as from the vectorial line equation using the depth label as parameter.
	 * While depth labels < 0 or > 1 are possible in principle, they do not necessarily make sense.  
	 * @param detectorCoord
	 * @param depthLabel
	 * @return
	 */
	public PointND reconstruct(BranchPoint detectorCoord, double depthLabel){
		return reconstruct(detectorCoord.get2DPointND(),depthLabel);
	}
	
	public Projection getProjectionMatrix(){
		return this.pMat;
	}
	
	public Grid2D getProjection(){
		return this.projection;
	}
	
	public void setProjection(Grid2D p){
		this.projection = p;
	}

	public Skeleton getSkeleton(){
		return this.skel;
	}
	
	public double getPrimaryAngle(){
		return this.primAngle;
	}
	
	/**
	 * Transforms a vector into its homogeneous counterpart
	 * @param p
	 * @return
	 */
	private PointND toHomogeneous(PointND p){
		int n = p.getDimension();
		double[] h = new double[n+1];
		for(int i = 0; i < n; i++){
			h[i] = p.get(i);
		}
		h[n] = 1;
		return new PointND(h);
	}
	
	/**
	 * Transforms a homogeneous vector/point into its canonical counter part, dividing by the last element such that 
	 * the last element == 1, and then omitting it.
	 * @param h
	 * @return
	 */
	private PointND toCanonical(PointND h){
		int n = h.getDimension()-1;
		double[] c = new double[n];
		for(int i = 0; i < n; i++){
			c[i] = h.get(i)/h.get(n);
		}
		return new PointND(c);
	}

	public KDTree2D<ComparablePoint2D> getKdTree() {
		return kdTree;
	}

	public double getDepthLabelRange() {
		return depthLabelRange;
	}

	public double getSourceToDetectorDistance(){
		return this.sourceDetectorDist;
	}
	
	public void setDepthLabelRange(double depthLabelRange) {
		this.depthLabelRange = depthLabelRange;
	}

	public double getDepthLabelCenterOffset() {
		return depthLabelCenterOffset;
	}

	public void setDepthLabelCenterOffset(double depthLabelCenterOffset) {
		this.depthLabelCenterOffset = depthLabelCenterOffset;
	}

	public ArrayList<BranchPoint> getBranchPointsAsList(){
		ArrayList<BranchPoint> bp = new ArrayList<BranchPoint>();
		for(int i = 0; i < this.skel.size(); i++){
			SkeletonBranch b = skel.get(i);
			int n = b.size();
			for(int j = 0; j < n; j++){
				bp.add(b.get(j));
			}
		}
		return bp;
	}
	
	public SimpleVector getCameraCenter(){
		return this.pMat.computeCameraCenter();
	}
	
}

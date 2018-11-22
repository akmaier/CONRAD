/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.skeleton;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.BranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Edge;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Graph;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonBranch;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonInfo;

public class Skeleton extends ArrayList<SkeletonBranch>{
	
	private static final long serialVersionUID = 4273079499645896557L;
	
	/** end point flag */
	public static byte END_POINT = 30;
	/** junction flag */
	public static byte JUNCTION = 70;
	/** slab flag */
	public static byte SLAB = 127;
	
	private double[] mean;
	private double[] spread;
	
	//===============================================================
	//	Methods
	//===============================================================
	
	private double minLength = 8;//pixel-units
	
	public Skeleton(){
		super();
	}
	
	public Skeleton(double lengthPruningThreshold){
		this.minLength = lengthPruningThreshold;
	}
	
	public void update(){
		calculateMeanAndSpread();
	}
	
	public void run(SkeletonInfo skInfo){
		ArrayList<Point> junc = skInfo.getJunctionPoints();
		ArrayList<Point> end = skInfo.getEndPoints();
		Graph[] graphs = skInfo.getGraphs();
		// first branch contains end and junction points
		SkeletonBranch b = new SkeletonBranch();
		if(this.minLength == 0){
			for(int i = 0; i < junc.size(); i++){
				b.add(new BranchPoint(junc.get(i), false, true));
			}
			for(int i = 0; i < end.size(); i++){
				b.add(new BranchPoint(end.get(i), true, false));
			}
		}
		// remaining branches contain actual branches
		for(int k = 0;  k < graphs.length; k++){
			Graph g = graphs[k];
			for(int j = 0; j < g.getEdges().size(); j++){
				Edge e = g.getEdges().get(j);
				if( (e.getLength() >= this.minLength) && (e.getSlabs().size() >= 0) ){
					this.add(new SkeletonBranch(e));
				}
			}
		}
		this.add(b);
	}
	
	/**
	 * Creates a list of branches from all edges in the graph.
	 * @param graphs
	 * @param junctions
	 */
	public void run(Graph[] graphs, ArrayList<Point> junctions){
		// calculate SkeletonBranches and add them to the list
		for(int i = 0; i < graphs.length; i++){
			Graph g = graphs[i];
			for(int j = 0; j < g.getEdges().size(); j++){
				Edge e = g.getEdges().get(j);
				if( (e.getLength() >= this.minLength) && (e.getSlabs().size() > 0) ){
					this.add(new SkeletonBranch(e, junctions));
				}
			}
		}
		calculateMeanAndSpread();
	}
	
	public double[] getSpread(){
		return this.spread;
	}
	
	/**
	 * Calculates the mean and "spread" (which is equal to the standard deviation) for the whole skeleton.
	 * This can be used to estimate "best viewing angles", as a high spread may indicate a view where there is 
	 * little overlap of distinct artery tree branches. 
	 */
	private void calculateMeanAndSpread(){
		int nBranches = this.size();
		int dim = 3;
		this.mean = new double[dim];
		this.spread = new double[dim];
		
		// calculate center of skeleton
		int nTotal = 0;
		for(int i = 0; i < nBranches; i++){
			SkeletonBranch b = this.get(i);
			int nPoints = b.size();
			nTotal += nPoints;
			for(int j = 0; j < nPoints; j++){
				mean[0] += b.get(j).x;
				mean[1] += b.get(j).y;
				mean[2] += b.get(j).z;
			}
		}
		for(int i = 0; i < dim; i++){
			mean[i] /= nTotal;
		}
		// calculate the "spread" which is equal to the standard deviation of the skeleton
		for(int i = 0; i < nBranches; i++){
			SkeletonBranch b = this.get(i);
			int nPoints = b.size();
			nTotal += nPoints;
			for(int j = 0; j < nPoints; j++){
				spread[0] += Math.pow((b.get(j).x - mean[0]),2);
				spread[1] += Math.pow((b.get(j).y - mean[1]),2);
				spread[2] += Math.pow((b.get(j).z - mean[2]),2);
			}
		}
		for(int i = 0; i < dim; i++){
			spread[i] = Math.sqrt(spread[i]/(nTotal-1));
		}
	}
	
	public ImageStack visualize(ImagePlus imp, boolean SHOW){
		return visualize(imp,this,SHOW);
	}
	
	public ImageStack visualize(ImagePlus imp, ArrayList<SkeletonBranch> list, boolean SHOW){		
		ImageStack stack = ImageStack.create(imp.getWidth(), imp.getHeight(), imp.getStackSize(), 8);
		for(int i = 0; i < list.size(); i++){
			SkeletonBranch b = list.get(i);
			for(int j = 0; j < b.size(); j++){
				BranchPoint p = b.get(j);
				if(!p.isEND()){
					stack.setVoxel(p.x, p.y, p.z, SLAB);
				}else if(p.isEND() && !p.isJUNCTION()){
					stack.setVoxel(p.x, p.y, p.z, END_POINT);
				}else{
					stack.setVoxel(p.x, p.y, p.z, JUNCTION);
				}
			}
		}
		if(SHOW){
			ImagePlus tagged = new ImagePlus("Skeleton-Value tagged Image");
			tagged.setStack(stack);
			tagged.show();
			IJ.run(tagged, "Fire", null);
		}
		return stack;
	}

	public void setMinLength(double l){
		this.minLength = l;
	}
	
}

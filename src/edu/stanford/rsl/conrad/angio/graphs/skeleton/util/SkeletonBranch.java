package edu.stanford.rsl.conrad.angio.graphs.skeleton.util;

import java.util.ArrayList;


public class SkeletonBranch extends ArrayList<BranchPoint>{
	
	private static final long serialVersionUID = -839966671066921317L;

	public SkeletonBranch(Edge e, ArrayList<Point> junctions){
		init(e, junctions);
	}	
	
	public SkeletonBranch(Edge e){
		init(e);
	}
	
	public SkeletonBranch(){
		super();
	}
	
	private void init(Edge e, ArrayList<Point> junctions){
		int nSlabs = e.getSlabs().size();
		
		ArrayList<Point> pointsV1 = e.getV1().getPoints();
		Point firstSlab = e.getSlabs().get(0);
		ArrayList<Point> pointsV2 = e.getV2().getPoints();
		Point lastSlab = e.getSlabs().get(nSlabs-1);
				
		for(int i = -1; i < nSlabs + 1; i++){
			BranchPoint p = null;
			if(i == -1){
				p = getClosestVertex(firstSlab, pointsV1);
				p = isJunction(p, junctions);
			}else if(i == nSlabs){
				p = getClosestVertex(lastSlab, pointsV2);
				p = isJunction(p, junctions);
			}else{
				// default for END and JUNCTION is false
				p = new BranchPoint(e.getSlabs().get(i));
			}
			this.add(p);
		}
	}
	
	private void init(Edge e){
		int nSlabs = e.getSlabs().size();
		
		for(int i = 0; i < nSlabs; i++){
			BranchPoint p = new BranchPoint(e.getSlabs().get(i));
			this.add(p);
		}
	}
	
	private BranchPoint isJunction(BranchPoint p, ArrayList<Point> junctions){
		for(int i = 0; i < junctions.size(); i++){
			if( p.equals(junctions.get(i)) ){
				return new BranchPoint(p, true, true);
			}
		}
		return p;
	}
	
	/**
	 * Determines the point in the list closest to the point of interest.
	 * In this context it is assumed that we are dealing with vertex points of skeleton branches.
	 * Therefore, the returned point will be an end point, but not necassarily a junction point.
	 * This has to be checked somewhere else. 
	 * @param p1
	 * @param pts
	 * @return
	 */
	private BranchPoint getClosestVertex(Point p1, ArrayList<Point> pts){
		double maxDist = Double.MAX_VALUE;
		int index = 0;
		for(int i = 0; i < pts.size(); i++){
			Point p2 = pts.get(i);
			double d = p1.distance(p2);
			if(maxDist > d){
				maxDist = d;
				index = i;
			}
		}
		return new BranchPoint(pts.get(index), true, false);
	}
	
	public BranchPoint getFirst(){
		return this.get(0);
	}
	public BranchPoint getLast(){
		return this.get(size() - 1);
	}
	
}

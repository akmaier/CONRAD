package edu.stanford.rsl.conrad.angio.graphs.connectedness;

import ij.IJ;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.util.HierarchyOnMST;
import edu.stanford.rsl.conrad.angio.util.io.PointAndRadiusIO;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;

public class MinSpanTreeReprojection {
	private int numVertices;
	private int[] parents = null;
	
	private double distanceThreshold = Double.MAX_VALUE;
	
	private Projection[] pMats;
	private Grid3D projections;
	
	private ArrayList<PointND> points = null;
	private ArrayList<MstEdge> edges = null;
	private ArrayList<MstEdge> mstEdges = null;
	
	public static void main(String[] args){
		
		String dir = ".../"; 
		Projection[] ps = ProjMatIO.readProjMats(dir+"pMat.txt");
		PointAndRadiusIO prio = new PointAndRadiusIO();
		prio.read(dir+"pts.txt");
		ArrayList<PointND> pts = prio.getPoints();
		Grid3D projs = ImageUtil.wrapImagePlus(IJ.openImage(dir+"hyst.tif"));
		
		MinSpanTreeReprojection mstRepro = new MinSpanTreeReprojection(ps,projs,pts, 1.0d);
		mstRepro.run();
		ArrayList<ArrayList<PointND>> con = mstRepro.getConnectedComponents();
		ArrayList<PointND> refinedPts = mstRepro.removeSmallComponents(con, 10);
		
		mstRepro = new MinSpanTreeReprojection(ps,projs,refinedPts, 2.0d);
		mstRepro.run();
		con = mstRepro.getConnectedComponents();
		con = mstRepro.removeSmallComponentsReturnList(con, 50);		
	}
		
	public MinSpanTreeReprojection(Projection[] pMats, Grid3D projs, ArrayList<PointND> pts, double threshold){
		this.pMats = pMats;
		this.projections = projs;
		this.distanceThreshold = threshold;
		this.points = pts;
		this.numVertices = pts.size();
		this.edges = new ArrayList<MstEdge>();
		this.mstEdges = new ArrayList<MstEdge>(numVertices-1);
	}
	
	public void run(){
		// mst stuff
		initializeEdges();
		runKruskalInternal();
	}
		
	private void runKruskalInternal(){
		// sort the edge list
		Collections.sort(edges);
		  
		UnionFind uf=new UnionFind(numVertices);
		  
		// Iterating over the sorted input edgeList
		for(int i = 0; i < edges.size(); i++){
			MstEdge edge = edges.get(i);
			int v1 = uf.find(edge.src);  //parent vertex for source
			int v2 = uf.find(edge.dest); //parent vertex for destination
			         
			// if parents do not match, consider edge for MST and, union the two vertices
			if(v1 != v2){
				mstEdges.add(edge);
				uf.Union(v1, v2);
			}
		}
		int[] pred = new int[numVertices];
		for(int i = 0; i < numVertices; i++){
			pred[i] = uf.getHolderAtIdx(i);
		}
		this.parents = pred;
	}
	
	/**
	 * Determines the set of connected components in the graph using the parent nodes. 
	 * @return
	 */
	public ArrayList<ArrayList<PointND>> getConnectedComponents(){
		ArrayList<ArrayList<PointND>> connected = new ArrayList<ArrayList<PointND>>();
		ArrayList<Integer> numUniqueStartNodes = new ArrayList<Integer>();
		for(int i = 0; i < parents.length; i++){
			if(!numUniqueStartNodes.contains(parents[i])){
				numUniqueStartNodes.add(parents[i]);
			}
		}
		for(int i = 0; i < numUniqueStartNodes.size(); i++){
			connected.add(new ArrayList<PointND>());
		}
		for(int i = 0; i < points.size(); i++){
			int parent = parents[i];
			for(int j = 0; j < numUniqueStartNodes.size(); j++){
				if(parent == numUniqueStartNodes.get(j)){
					connected.get(j).add(points.get(i));
					continue;
				}
			}
		}
		
		return connected;
	}
	
	public ArrayList<PointND> removeSmallComponents(ArrayList<ArrayList<PointND>> comp, int componentSize){
		ArrayList<PointND> ptsPruned = new ArrayList<PointND>();
		int numAdded = 0;
		for(int i = 0; i < comp.size(); i++){
			if(comp.get(i).size() > componentSize){
				ptsPruned.addAll(comp.get(i));
				numAdded++;
			}
		}
		System.out.println("Removed "+String.valueOf(comp.size()-numAdded)+
				" components of "+String.valueOf(comp.size())+" components.");
		return ptsPruned;
	}
	
	public ArrayList<ArrayList<PointND>> removeSmallComponentsReturnList(ArrayList<ArrayList<PointND>> comp, int componentSize){
		ArrayList<ArrayList<PointND>> ptsPruned = new ArrayList<ArrayList<PointND>>();
		int numAdded = 0;
		for(int i = 0; i < comp.size(); i++){
			if(comp.get(i).size() > componentSize){
				ptsPruned.add(comp.get(i));
				numAdded++;
			}
		}
		System.out.println("Removed "+String.valueOf(comp.size()-numAdded)+
				" components of "+String.valueOf(comp.size())+" components.");
		return ptsPruned;
	}
	
	public ArrayList<PointND> getLargestComponents(ArrayList<ArrayList<PointND>> comp){
		int largestSize = 0;
		int idx = 0;
		for(int i = 0; i < comp.size(); i++){
			if(comp.get(i).size() > largestSize){
				largestSize = comp.get(i).size();
				idx = i;
			}
		}
		return comp.get(idx);
	}
		
	public ArrayList<Edge> getMSTasEdgeList(){
		ArrayList<Edge> edges = 
				new ArrayList<Edge>();
		ArrayList<int[]> connections = getMSTconnectivitiy();
		for(int i = 0; i < connections.size(); i++){
			int[] c = connections.get(i);
			edges.add(new Edge(points.get(c[0]),points.get(c[1])));
		}
		return edges;
	}
		
	public ArrayList<ArrayList<Edge>> getMstHierarchical(){
		HierarchyOnMST hierMst = new HierarchyOnMST();
		hierMst.run(points, getMSTconnectivitiy(), 0.0);
		ArrayList<ArrayList<Edge>> hierarch = hierMst.getVesselTreeAsEdgeListSep();
//		ArrayList<Edge> test = new ArrayList<Edge>();
//		ArrayList<int[]> con = getMSTconnectivitiy();
//		for(int i = 0; i < con.size(); i++){
//			test.add(new Edge(points.get(con.get(i)[0]),points.get(con.get(i)[1])));
//		}
//		ArrayList<ArrayList<Edge>> test2 = new ArrayList<ArrayList<Edge>>();
//		test2.add(test);
//		EdgeViewer.renderEdgesComponents(hierarch);
		return hierarch;
	}
	
		
	public PointND calculateWeightedCenter(ArrayList<ArrayList<PointND>> comp){
		return calculateWeightedCenter(comp, 0);
	}
	
	public PointND calculateWeightedCenter(ArrayList<ArrayList<PointND>> comp, int componentSize){
		ArrayList<SimpleVector> centers = new ArrayList<SimpleVector>();
		ArrayList<Integer> sizes = new ArrayList<Integer>();
		double totSize = 0;
		for(int i = 0; i < comp.size(); i++){
			int compSize = comp.get(i).size();
			if(compSize > componentSize){
				SimpleVector cent = new SimpleVector(comp.get(0).get(0).getDimension());
				for(int j = 0; j < comp.get(i).size(); j++){
					cent.add(comp.get(i).get(j).getAbstractVector());
				}
				cent.divideBy(compSize);
				centers.add(cent);
				sizes.add(compSize);
				totSize += compSize;
			}			
		}
		SimpleVector cent = centers.get(0);
		cent.multipliedBy(sizes.get(0));
		for(int i = 1; i < centers.size(); i++){
			cent.add(centers.get(i).multipliedBy(sizes.get(i)));
		}
		cent.divideBy(totSize);
		return new PointND(cent.copyAsDoubleArray());
	}
		
	/**
	 * Initialized the edges. Assumes an undirected graph.
	 */
	private void initializeEdges(){
		System.out.println("Initializing edgs using reprojection. This may take some time.");
		for(int i = 0; i < points.size(); i++){
			// saves the 8 octants in 3D space
			ArrayList<ArrayList<Integer>> octants = new ArrayList<ArrayList<Integer>>();
			for(int o = 0; o < 8; o++){
				octants.add(new ArrayList<Integer>());
			}
			for(int j = 0; j < points.size(); j++){				
				if(j == i){
					continue;
				}
				double cost = points.get(i).euclideanDistance(points.get(j));
				if(cost < distanceThreshold){
					SimpleVector v = new SimpleVector(points.get(j).getAbstractVector().clone());
					v.subtract(points.get(i).getAbstractVector());
					v.normalizeL2();
					int oct = getOctant(v);
					octants.get(oct).add(j);		 
				}				
			}
				
			for(int k = 0; k < octants.size(); k++){
				int minEdgeIdx = -1;
				double minCost = Double.MAX_VALUE;
				for(int j = 0; j < octants.get(k).size(); j++){
					int targetNodeIdx = octants.get(k).get(j);
					double cost = computeCost(points.get(i), points.get(targetNodeIdx));
					if(cost < minCost){
						minCost = cost;
						minEdgeIdx = targetNodeIdx;
					}
				}
				if(minEdgeIdx >= 0 && minEdgeIdx > i){
					MstEdge edge = new MstEdge(i, minEdgeIdx, minCost);
					edges.add(edge);
				}
				
			}						
		}
	}
	
	public ArrayList<MstEdge> getMST(){
		return this.mstEdges;
	}
	
	private double computeCost(PointND p1, PointND p2) {		
		double sampling = projections.getSpacing()[0]/5f;
		double[] vals = new double[pMats.length];
		double norm = 0;
		for(int i = 0; i < pMats.length; i++){
			SimpleVector dirCentPoint = new SimpleVector(p1.getCoordinates());
			dirCentPoint.subtract(pMats[i].computeCameraCenter());
			dirCentPoint.normalizeL2();
			SimpleVector edgeDir = new SimpleVector(p2.getCoordinates());
			edgeDir.subtract(p1.getAbstractVector());
			edgeDir.normalizeL2();
			double foreshortening = (1-SimpleOperators.multiplyInnerProd(dirCentPoint, edgeDir));
			norm += foreshortening;
			
			SimpleVector det1 = new SimpleVector(2);
			pMats[i].project(p1.getAbstractVector(), det1);
			SimpleVector det2 = new SimpleVector(2);
			pMats[i].project(p2.getAbstractVector(), det2);
			SimpleVector dir = det2.clone();
			dir.subtract(det1);
			double length = dir.normL2();
			dir.normalizeL2();
			int numSamples = (int)Math.floor(length / sampling);
			float val = 0;
			for(int j = 0; j < numSamples; j++){
				det1.add(dir.multipliedBy(j*sampling));
				val += InterpolationOperators.interpolateLinear(
						projections.getSubGrid(i), det1.getElement(0), det1.getElement(1));	
			}
			val += InterpolationOperators.interpolateLinear(
					projections.getSubGrid(i), det2.getElement(0), det2.getElement(1));
			vals[i] = val * foreshortening;
		}
		Arrays.sort(vals);
		double penalty = 0;
		for(int i = 0; i < pMats.length; i++){
			penalty += vals[i];
		}
		penalty /= norm;
		double dist = p1.euclideanDistance(p2);
		return penalty*dist; // vals[pMats.length-1];//
	}
	
	public ArrayList<int[]> getMSTconnectivitiy(){
		ArrayList<int[]> con = new ArrayList<int[]>();
		for(int i = 0; i < mstEdges.size(); i++){
			MstEdge e = mstEdges.get(i);
			int[] c = new int[]{e.src, e.dest};
			con.add(c);
		}
		return con;
	}
	
	public class UnionFind {
		// Node Holder having UFNode
		private UFNode[] nodeHolder;
		
		// number of node
		private int count;
		
		public UnionFind(int size) {
			if (size < 0){
				throw new IllegalArgumentException();
			}
			count = size;
			nodeHolder = new UFNode[size];
			for (int i = 0; i < size; i++) {
				// default values, node points to itself and rank is 1
				nodeHolder[i] = new UFNode(i, 1); 
			}
		}
		
		public int getHolderAtIdx(int idx){
			return nodeHolder[idx].parent;
		}
		
		/**
		* Finds the parent of a given vertex, using recursion
		* 
		* @param vertex
		* @return
		*/
		public int find(int vertex) {
			if (vertex < 0 || vertex >= nodeHolder.length){
				throw new IndexOutOfBoundsException();
			}
			if (nodeHolder[vertex].parent != vertex){
				nodeHolder[vertex].parent = find(nodeHolder[vertex].parent);
			}
			return nodeHolder[vertex].parent;
		}
		
		public int getCount() {
			return count;
		}
		
		/**
		* @param v1
		*            : vertex 1 of some cluster
		* @param v2
		*            : vertex 2 of some cluster
		* @return true if both vertex have same parent
		*/
		public boolean isConnected(int v1, int v2) {
			return find(v1) == find(v2);
		}
		
		/**
		* unions two cluster of two vertices
		* @param v1
		* @param v2
		*/
		public void Union(int v1, int v2) {
			int i = find(v1);
			int j = find(v2);
		
			if (i == j){
				return;
			}
		
			if (nodeHolder[i].rank < nodeHolder[j].rank) {
				nodeHolder[i].parent = j;
				nodeHolder[j].rank = nodeHolder[j].rank + nodeHolder[i].rank;
			} else {
				nodeHolder[j].parent = i;
				nodeHolder[i].rank = nodeHolder[i].rank + nodeHolder[j].rank;
			}
			count--;
		}
		
	}
	
	class UFNode {
		int parent; // parent of Vertex at i in the nodeHolder
		int rank; // Number of object present in the tree/ Cluster
		
		UFNode(int parent, int rank) {
			this.parent = parent;
			this.rank = rank;
		}
	}

	class MstEdge implements Comparable<MstEdge>{
		 int src;
		 int dest;
		 double weight;

		 public MstEdge(int src,int dest,double weight) {
		  this.src=src;
		  this.dest=dest;
		  this.weight=weight;
		 }

		 @Override
		 public String toString() {
		  return "Edge: "+src+ " - " + dest + " | " +"  Weight: "+ weight;
		 }

		 @Override
		 public int compareTo(MstEdge another) {
			 return Double.valueOf(this.weight).compareTo(Double.valueOf(another.weight));
		 }
	}
	
	private int getOctant(SimpleVector v){
		// z positive: 1-4
		if(v.getElement(2) > 0){
			// check y
			if(v.getElement(1) > 0){
				// check x
				if(v.getElement(0) > 0){
					return 0;
				}else{
					return 1;
				}
			}else{
				// check x
				if(v.getElement(0) > 0){
					return 3;
				}else{
					return 2;
				}
			}
		}else{
			// check y
			if(v.getElement(1) > 0){
				// check x
				if(v.getElement(0) > 0){
					return 4;
				}else{
					return 5;
				}
			}else{
				// check x
				if(v.getElement(0) > 0){
					return 7;
				}else{
					return 6;
				}
			}
		}
	}
}
	



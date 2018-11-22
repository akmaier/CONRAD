package edu.stanford.rsl.conrad.angio.graphs.connectedness;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.util.HierarchyOnMST;

public class MinimumSpanningTree{

	private int numVertices;
	private int[] parents = null;
	
	private double distanceThreshold = Double.MAX_VALUE;
	
	private ArrayList<PointND> points = null;
	private ArrayList<MstEdge> edges = null;
	private ArrayList<MstEdge> mstEdges = null;
	
	public static void main(String[] args){
		ArrayList<PointND> pts = new ArrayList<PointND>();
		pts.add(new PointND(0,0));
		pts.add(new PointND(1,0));
		pts.add(new PointND(2,0));
		pts.add(new PointND(1,1));
		pts.add(new PointND(0,2));
		pts.add(new PointND(0,1));
		pts.add(new PointND(2,2));
		pts.add(new PointND(5,5));
		pts.add(new PointND(5,6));
		
		MinimumSpanningTree mts = new MinimumSpanningTree(pts, 2.0d);
		mts.run(true);
		ArrayList<ArrayList<PointND>> con = mts.getConnectedComponents();
		ArrayList<int[]> connectedness = mts.getConnectedness(con.get(0));
		System.out.println("Numer of connections: "+connectedness.size());
		
	}
	
	public MinimumSpanningTree(ArrayList<PointND> pts){
		this.points = pts;
		this.numVertices = pts.size();
		this.edges = new ArrayList<MstEdge>();
		this.mstEdges = new ArrayList<MstEdge>(numVertices-1);
	}
	
	public MinimumSpanningTree(ArrayList<PointND> pts, double threshold){
		this.distanceThreshold = threshold;
		this.points = pts;
		this.numVertices = pts.size();
		this.edges = new ArrayList<MstEdge>();
		this.mstEdges = new ArrayList<MstEdge>(numVertices-1);
	}
	
	public void run(){
		initializeEdges();
		runKruskalInternal(false);
	}
	
	public void run(boolean printTree){
		initializeEdges();
		runKruskalInternal(printTree);
	}
	
	private void runKruskalInternal(boolean printTree){
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
		
		// print the final MST
		if(printTree){
			printKruskalEdges();
			printParents();
		}
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
	
	public ArrayList<PointND> getLargestConnectedComponent(){
		ArrayList<ArrayList<PointND>> cc = getConnectedComponents();
		int idx = 0;
		int maxSize = 0;
		for(int i = 0; i < cc.size(); i++){
			if(cc.get(i).size()>maxSize){
				maxSize = cc.get(i).size();
				idx = i;
			}
		}
		return cc.get(idx);
	}
	
	public ArrayList<PointND> removeSmallComponents(ArrayList<ArrayList<PointND>> comp, int componentSize){
		ArrayList<PointND> ptsPruned = new ArrayList<PointND>();
		for(int i = 0; i < comp.size(); i++){
			if(comp.get(i).size() > componentSize){
				ptsPruned.addAll(comp.get(i));
			}
		}
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
	
	public ArrayList<ArrayList<PointND>> getLargestComponents(ArrayList<ArrayList<PointND>> comp, int num){
		int[] sizes = new int[comp.size()];
		Integer[] idx = new Integer[comp.size()];
		for(int i = 0; i < comp.size(); i++){
			sizes[i] = comp.get(i).size();
			idx[i] = i;
		}
		Arrays.sort(idx, (o1,o2) -> Integer.compare(sizes[o1], sizes[o2]));
		ArrayList<ArrayList<PointND>> largestComps = new ArrayList<ArrayList<PointND>>();
		for(int i = 0; i < num; i++){
			if(i < comp.size()){
				largestComps.add(comp.get(idx[comp.size()-1-i]));
			}
		}
		return largestComps;
	}
	
	public Grid2D visualize(Grid2D img, ArrayList<PointND> comp){
		Grid2D plot = new Grid2D(img.getSize()[0], img.getSize()[1]);
		plot.setSpacing(img.getSpacing());
		plot.setOrigin(img.getOrigin());
		for(int i = 0; i < comp.size(); i++){
			int iX = (int)comp.get(i).get(0);
			int iY = (int)comp.get(i).get(1);
			plot.setAtIndex(iX, iY, 1);
		}
		return plot;
	}
	
	public ArrayList<int[]> getConnectedness(ArrayList<PointND> points){
		ArrayList<MstEdge> edges = new ArrayList<MstEdge>();
		int numVertices = points.size();
		for(int i = 0; i < numVertices; i++){
			PointND src = points.get(i);
			for(int j = i+1; j < numVertices; j++){
				double eucDist = src.euclideanDistance(points.get(j));
				if(eucDist < distanceThreshold){
					edges.add(new MstEdge(i,j,eucDist));
				}
			}
		}
		ArrayList<MstEdge> mstEdges = new ArrayList<MstEdge>();
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
		ArrayList<int[]> con = new ArrayList<int[]>();
		for(int i = 0; i < mstEdges.size(); i++){
			MstEdge e = mstEdges.get(i);
			int[] c = new int[]{e.src, e.dest};
			con.add(c);
		}
		return con;
	}
	
	public ArrayList<PointND> computeEndPoints(ArrayList<PointND> pts){
		ArrayList<int[]> con = getConnectedness(pts);
		int[] counts = new int[pts.size()];
		for(int i = 0; i < con.size(); i++){
			int[] c = con.get(i);
			counts[c[0]] += 1;
			counts[c[1]] += 1;
		}
		ArrayList<PointND> endPts = new ArrayList<PointND>();
		for(int i = 0; i < counts.length; i++){
			if(counts[i] == 1){
				endPts.add(pts.get(i));
			}
		}
		return endPts;
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
	
	public ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> getMSTasEdgeList(ArrayList<PointND> points){
		ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> edges = 
				new ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>();
		ArrayList<int[]> connections = getConnectedness(points);
		for(int i = 0; i < connections.size(); i++){
			int[] c = connections.get(i);
			edges.add(new edu.stanford.rsl.conrad.geometry.shapes.simple.Edge(points.get(c[0]),points.get(c[1])));
		}
		return edges;
	}
	
	/**
	  * Printing the Kruskal MST edges
	  */
	 private void printKruskalEdges(){
		 for(MstEdge edge:mstEdges){
			 System.out.println(edge);
		 }
	 }
	 
	 private void printParents(){
		 String prnts = "Parents: ";
		 for(int i = 0; i < parents.length; i++){
			 prnts += " "+String.valueOf(parents[i]);
		 }
		 System.out.println(prnts);
	 }
	
	/**
	 * Initialized the edges. Assumes an undirected graph.
	 */
	private void initializeEdges(){
		for(int i = 0; i < numVertices; i++){
			PointND src = points.get(i);
			for(int j = i+1; j < numVertices; j++){
				double eucDist = src.euclideanDistance(points.get(j));
				if(eucDist < distanceThreshold){
					edges.add(new MstEdge(i,j,eucDist));
				}
			}
		}
	}
	
	public ArrayList<ArrayList<Edge>> getMstHierarchical(){
		HierarchyOnMST hierMst = new HierarchyOnMST();
		hierMst.run(points, getMSTconnectivitiy(), 0.0);
		ArrayList<ArrayList<Edge>> hierarch = hierMst.getVesselTreeAsEdgeListSep();
		//EdgeViewer.renderEdgesComponents(hierarch);
		return hierarch;
	}
	
	public ArrayList<ArrayList<Edge>> getMstHierarchical(int startNodeIdx){
		HierarchyOnMST hierMst = new HierarchyOnMST();
		hierMst.run(points, getMSTconnectivitiy(), 0.0, startNodeIdx);
		ArrayList<ArrayList<Edge>> hierarch = hierMst.getVesselTreeAsEdgeListSep();
		//EdgeViewer.renderEdgesComponents(hierarch);
		return hierarch;
	}
	
	public ArrayList<MstEdge> getMST(){
		return this.mstEdges;
	}
	
	public ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> getMSTasEdgeList(){
		ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> edges = 
				new ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>();
		ArrayList<int[]> connections = getMSTconnectivitiy();
		for(int i = 0; i < connections.size(); i++){
			int[] c = connections.get(i);
			edges.add(new edu.stanford.rsl.conrad.geometry.shapes.simple.Edge(points.get(c[0]),points.get(c[1])));
		}
		return edges;
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
}
	



/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness;

import ij.IJ;
import ij.ImageJ;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.PriorityQueue;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Edge;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Graph;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Node;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranch;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.SkeletonUtil;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;
import edu.stanford.rsl.conrad.angio.points.DistanceTransform2D;
import edu.stanford.rsl.conrad.angio.util.image.ImageOps;

public class Dijkstra2D {
	
	private boolean verbose = true;
	
	public enum EightConnectedLattice{
		p1p0 (new int[]{1,0,0}),
		p1p1 (new int[]{1,1,1}),
		p0p1 (new int[]{0,1,2}),
		m1p1 (new int[]{-1,1,3}),
		m1m0 (new int[]{-1,0,4}),
		m1m1 (new int[]{-1,-1,5}),
		m0m1 (new int[]{0,-1,6}),
		p1m1 (new int[]{1,-1,7});
		
		private int[] offs = new int[]{0,0};
		
		EightConnectedLattice(int[] offs){
			this.offs = offs;
		}
		
		public int[] getShift(){
			return this.offs;
		}		
	}
	
	private Graph graph = null;
	private VesselTree vessels = null;
	
	private double pruningLength = 20;
	
	private Grid2D costMap = null;
	private Grid2D radii = null;
	private Point startPoint = null;
	private double costThreshold = 0;
	private double endNodeThreshold = 0;
	private double averageCost = -1;
	
	ArrayList<Point> endPoints = null;
	
	
	public static void main(String[] args){
		String dir = ".../"; 
		Grid3D mstImg = ImageUtil.wrapImagePlus(IJ.openImage(dir+"mst.tif"));
		int idx = 2;
		
		ArrayList<PointND> pts = ImageOps.thresholdedPointList(mstImg.getSubGrid(idx), 0.5);
		Point startPoint = SkeletonUtil.determineStartPoint(pts);
		// determine end point candidates
		ArrayList<Point> endPts = new ArrayList<Point>();
		for(int i = 0; i < pts.size(); i++){
			endPts.add(new Point((int)pts.get(i).get(0),(int)pts.get(i).get(1),0));
		}
		// calculate distance transform from the centerline candidates
		DistanceTransform2D distTrafo = new DistanceTransform2D(mstImg.getSubGrid(idx), pts, true);
		Grid2D cm = distTrafo.run();
		
		// extract allowed paths connecting end point candidates and start node
		Dijkstra2D dijkstra = new Dijkstra2D();
		dijkstra.setPruningLength(2.5);
		dijkstra.run(cm, cm, startPoint, 5, endPts);
		Grid2D vtImage = dijkstra.visualizeVesselTree(dijkstra.getVesselTree());
		
		new ImageJ();
		vtImage.show();
		
	}
	
	/**
	 * Using this method assumes that the complete cost map has already been calculated and is represented by g.
	 * @param g
	 * @param radii
	 * @param sp
	 * @param cth
	 */
	public void run(Grid2D g, Grid2D radii, Point sp, double cth){
		this.costMap = g;
		this.radii = radii;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.endNodeThreshold = cth;
		this.graph = getGraph();
		
//		Grid2D graphImg = visualizeGraphNodes(graph); 
//		graphImg.show();
		
		this.vessels = extractCenterlineFromGraph();
	}
	
	/**
	 * Using this method assumes that the complete cost map has already been calculated and is represented by g.
	 * @param g
	 * @param radii
	 * @param sp
	 * @param cth
	 * @param endNodeTh
	 */
	public void run(Grid2D g, Grid2D radii, Point sp, double cth, double endNodeTh){
		this.costMap = g;
		this.radii = radii;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.endNodeThreshold = endNodeTh;
		this.graph = getGraph();
		
		this.vessels = extractCenterlineFromGraph();
	}
	
	public void run(Grid2D g, Grid2D radii, Point sp, double cth, ArrayList<Point> endPts){
		this.costMap = g;
		this.radii = radii;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.endNodeThreshold = cth;
		this.endPoints = endPts;
		this.graph = getGraph();
		
		this.vessels = extractCenterlineFromGraphWithEndnodes();
	}
	
	public void runDifferentPoints(Point sp){
		this.startPoint = sp;

		this.vessels = extractCenterlineFromGraph();
	}
	
	public void runDifferentPoints(Point sp, ArrayList<Point> endPts){
		this.startPoint = sp;
		this.endPoints = endPts;
		
		this.vessels = extractCenterlineFromGraphWithEndnodes();
	}
	
	
	public VesselTree getVesselTree(){
		return this.vessels;
	}
	
	/**
	 * Essentially a Dijkstra's algorithm with backtracking.
	 * @return
	 */
	private VesselTree extractCenterlineFromGraph() {
		VesselTree tree = new VesselTree();
		Node root = null;// = new Node(startPoint);
		
		// extract nodes from graph
		ArrayList<Node> graphNodes = graph.getNodes();
		
		double minDist = Double.MAX_VALUE;
		for(int i = 0; i < graphNodes.size(); i++){
			Node n = graphNodes.get(i);
			double dist = Math.pow(n.x-startPoint.x,2) + Math.pow(n.y-startPoint.y, 2);
			if(dist < minDist && n.getEdges().size() > 0){
				minDist = dist;
				root = n;
			}
		}

		// create new priority queue with descending cost order
		PriorityQueue<Node> endNodes = new PriorityQueue<Node>(Collections.reverseOrder());

		// create new priority queue with ascending cost order
		PriorityQueue<Node> nodeQueue = new PriorityQueue<Node>();
		// set complete cost of root node to zero
		root.setCompleteCost(0);
		// put root node in queue
		nodeQueue.add(root);

		// Dijkstra computation of the graph
		if(verbose)
			System.out.println("Dijkstra computation");

		// treat all nodes in queue
		while (!nodeQueue.isEmpty()) {

			// counter for number of accepted neighboring nodes
			int counter = 0;
			// get the head of the queue
			Node n1 = nodeQueue.poll();
			// extract adjacent edges from current node
			ArrayList<Edge> adjacentEdges = n1.getEdges();

			// treat every edge at the current node
			int edgesBelowEndNodeTh = 0;
			for (int i = 0; i < adjacentEdges.size(); i++) {
				Edge e = adjacentEdges.get(i);

				Node n2 = graphNodes.get(e.getIdxN2());

				// compute the new complete cost to the target node with cost of current node plus the edge cost
				double edgeCost = e.getCost();
				double costViaCurEdge = n1.getCompleteCost() + edgeCost;
				
				if(edgeCost < endNodeThreshold){
					edgesBelowEndNodeTh++;
				}
				// check if new cost is below the already existing cost at target node
				if (costViaCurEdge < n2.getCompleteCost()) {
					// replace complete cost in target node
					n2.setCompleteCost(costViaCurEdge);
					// set the previous node for backtracking
					n2.setPreviousNode(n1);
					// set the edge to the previous node
					n2.setEdgeToPrevious(e);
					// put the target node into queue
					nodeQueue.add(n2);
					// increase the counter
					counter++;
				}
			}
			if (counter == 0) {
				// if the counter is zero check if there was at least one edge with cost below endNodeThreshold
				// the current node does not have any next low cost edge this point is an end point
				if(edgesBelowEndNodeTh != 0){
					endNodes.add(n1);
				}
			}
		}

		// perform backtracking on graph
		if(verbose)
			System.out.println("Backtracking");

		while (!endNodes.isEmpty()) {
			double completeLength = 0;
			double completeCost = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = endNodes.poll();

			branch.add(new VesselBranchPoint(cur.x, cur.y, cur.z, cur.getEdgeToPrevious().getRadius()));
			Node previous = cur.getPreviousNode();
			
			while (previous != null && !previous.isVisited()) {

				completeLength += cur.getEdgeToPrevious().getLength();
				completeCost += cur.getEdgeToPrevious().getCost();

				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}

				double[] physicalCoordinates = costMap.indexToPhysical(previous.x, previous.y);

				branch.add(new VesselBranchPoint(previous.x, previous.y, previous.z, physicalCoordinates[0], physicalCoordinates[1], 1, meanRadius));

				previous.setVisited();

				cur = previous;
				previous = previous.getPreviousNode();
			}

			branch.setLength(completeLength);
			branch.setCost(completeCost);

			if (completeLength > pruningLength) {
				if(averageCost > 0){
					double meanCost = completeCost / completeLength;
					if(meanCost < averageCost){
						tree.add(branch);
					}
				}else{
					tree.add(branch);
				}
			}

		}

		return tree;
	}
	
	/**
	 * Essentially a Dijkstra's algorithm with backtracking.
	 * @return
	 */
	private VesselTree extractCenterlineFromGraphWithEndnodes() {
		VesselTree tree = new VesselTree();
		Node root = null;// = new Node(startPoint);
		
		// extract nodes from graph
		ArrayList<Node> graphNodes = graph.getNodes();
		
		double minDist = Double.MAX_VALUE;
		for(int i = 0; i < graphNodes.size(); i++){
			Node n = graphNodes.get(i);
			double dist = Math.pow(n.x-startPoint.x,2) + Math.pow(n.y-startPoint.y, 2);
			if(dist < minDist){
				minDist = dist;
				root = n;
			}
		}

		// create new priority queue with descending cost order
		PriorityQueue<Node> endNodes = new PriorityQueue<Node>(Collections.reverseOrder());

		// create new priority queue with ascending cost order
		PriorityQueue<Node> nodeQueue = new PriorityQueue<Node>();
		// set complete cost of root node to zero
		root.setCompleteCost(0);
		// put root node in queue
		nodeQueue.add(root);

		// Dijkstra computation of the graph
		if(verbose)
			System.out.println("Dijkstra computation");

		// treat all nodes in queue
		while (!nodeQueue.isEmpty()) {

			// get the head of the queue
			Node n1 = nodeQueue.poll();
			// extract adjacent edges from current node
			ArrayList<Edge> adjacentEdges = n1.getEdges();

			// treat every edge at the current node
			for (int i = 0; i < adjacentEdges.size(); i++) {
				Edge e = adjacentEdges.get(i);

				// extract the target node of the current edge from node array
				// localize target node of edge in node array of graph
				Node n2 = graphNodes.get(e.getIdxN2());

				// compute the new complete cost to the target node with cost of current node plus the edge cost
				double edgeCost = e.getCost();
				double costViaCurEdge = n1.getCompleteCost() + edgeCost;
				
				// check if new cost is below the already existing cost at target node
				if (costViaCurEdge < n2.getCompleteCost()) {
					// replace complete cost in target node
					n2.setCompleteCost(costViaCurEdge);
					// set the previous node for backtracking
					n2.setPreviousNode(n1);
					// set the edge to the previous node
					n2.setEdgeToPrevious(e);
					// put the target node into queue
					nodeQueue.add(n2);
				}
			}
			if (isInEndNodes(n1)) {
				endNodes.add(n1);
			}
		}

		// perform backtracking on graph
		if(verbose)
			System.out.println("Backtracking");

		while (!endNodes.isEmpty()) {
			double completeLength = 0;
			double completeCost = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = endNodes.poll();

			branch.add(new VesselBranchPoint(cur.x, cur.y, cur.z, 1));
			Node previous = cur.getPreviousNode();
			
			while (previous != null && !previous.isVisited()) {

				completeLength += cur.getEdgeToPrevious().getLength();
				completeCost += cur.getEdgeToPrevious().getCost();

				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}

				double[] physicalCoordinates = costMap.indexToPhysical(previous.x, previous.y);

				branch.add(new VesselBranchPoint(previous.x, previous.y, previous.z, physicalCoordinates[0], physicalCoordinates[1], 1, meanRadius));

				previous.setVisited();

				cur = previous;
				previous = previous.getPreviousNode();
			}

			branch.setLength(completeLength);
			branch.setCost(completeCost);

			if (completeLength > pruningLength) {
				tree.add(branch);
			}

		}

		return tree;
	}
	
	private boolean isInEndNodes(Node n){
		for(int i = 0; i < endPoints.size(); i++){
			Point p = endPoints.get(i);
			if(p.x-n.x == 0){
				if(p.y-n.y == 0){
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * Compute an undirected graph with all nodes which edges are below the cost threshold.
	 * @param costMap 
	 * @param startPoint - start coordinates for tree should lie inside desired vessel tree
	 * @param medThreshold - medialness threshold
	 * @return graph
	 */
	private Graph getGraph() {
		if(verbose)
			System.out.println("Extracting graph.");

		// create graph
		Graph costGraph = new Graph();
		Rectangle rect = new Rectangle(1,1,costMap.getWidth()-1, costMap.getHeight()-1);
						
		for(int i = rect.x; i < rect.x+rect.width; i++){
			for(int j = rect.y; j < rect.y+rect.height; j++){
				Point sourcePoint = new Point(i,j,1);
				Node sourceNode = new Node(sourcePoint);
				costGraph.addNodeNoChecking(sourceNode);
			}	
		}
		
		int numEdges = 0;
		
		for(int i = rect.x+1; i < rect.x+rect.width-1; i++){
			int idxx = i-rect.x;
			for(int j = rect.y+1; j < rect.y+rect.height-1; j++){				
				int idxy = j-rect.y;
				
				int sourceNodeIdx = idxx*rect.height + idxy;
				for(EightConnectedLattice shift : EightConnectedLattice.values()){
					int[] s = shift.getShift();
					int targetNodeIdx = (idxx+s[0])*rect.height + (idxy+s[1]);
					float cost = costMap.getAtIndex(i+s[0], j+s[1]);					
					if(cost < costThreshold){						
						Edge edge = new Edge(new Node(i,j,1), sourceNodeIdx, new Node(i+s[0],j+s[1],1),targetNodeIdx, cost, radii.getAtIndex(i, j));
						costGraph.setEdgeToNode(sourceNodeIdx, edge);
						numEdges++;
					}
				}			
			}
		}
		if(verbose)
			System.out.println("Total number of edges in the graph: " + numEdges);
		return costGraph;
	}

	
	public Grid2D visualizeVesselTree(VesselTree v){
		int[] gSize = this.costMap.getSize();
		double[] gSpace = this.costMap.getSpacing();
		Grid2D g = new Grid2D(gSize[0], gSize[1]);
		g.setSpacing(gSpace[0], gSpace[1]);
		for(int i = 0; i < v.size(); i++){
			VesselBranch b = v.get(i);
			for(int j = 0; j < b.size(); j++){
				Point p = b.get(j);
				g.setAtIndex(p.x, p.y, 1);
			}
		}		
		return g;
	}
	
	public ArrayList<PointND> getVesselTreeAsList(){
		ArrayList<PointND> list = new ArrayList<PointND>();
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			for(int j = 0; j < vb.size(); j++){
				VesselBranchPoint bp = vb.get(j);
				list.add(new PointND(bp.getPhysCoordinates()));
			}
		}
		return list;
	}
	
	public Grid2D visualizeGraphNodes(Graph graph){
		int[] gSize = this.costMap.getSize();
		double[] gSpace = this.costMap.getSpacing();
		Grid2D g = new Grid2D(gSize[0], gSize[1]);
		g.setSpacing(gSpace[0], gSpace[1]);
		ArrayList<Node> nodes = graph.getNodes();
		for(int i = 0; i < nodes.size(); i++){
			Node n = nodes.get(i);
			g.setAtIndex(n.x, n.y, 1);
		}		
		return g;
	}

	public double getPruningLength() {
		return pruningLength;
	}

	public void setPruningLength(double pruningLength) {
		this.pruningLength = pruningLength;
	}
	
	public void setAllowableAverageCost(double avgCost) {
		this.averageCost = avgCost;
	}

	public boolean isVerbose() {
		return verbose;
	}

	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
	
}

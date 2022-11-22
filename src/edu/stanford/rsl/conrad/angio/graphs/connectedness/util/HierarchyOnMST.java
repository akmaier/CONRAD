/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness.util;

import java.util.ArrayList;
import java.util.PriorityQueue;

import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Edge;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Graph;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Node;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranch;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;

public class HierarchyOnMST {
	private Graph graph = null;
	private ArrayList<int[]> connections = null;
	private VesselTree vessels = null;
	
	private ArrayList<PointND> points = null;
	
	private double pruningLength = 20;
		
			
	public void run(ArrayList<PointND> pts, ArrayList<int[]> connections, double pruningLength){
		this.points = pts;
		this.connections = connections;
		this.graph = getGraph();
		this.pruningLength = pruningLength;	
		this.vessels = extractCenterlineFromGraph(-1);
	}
	
	public void run(ArrayList<PointND> pts, ArrayList<int[]> connections, double pruningLength, int startNodeIdx){
		this.points = pts;
		this.connections = connections;
		this.graph = getGraph();
		this.pruningLength = pruningLength;	
		this.vessels = extractCenterlineFromGraph(startNodeIdx);
	}
	
	public void run(ArrayList<PointND> pts, ArrayList<int[]> connections, int startNodeIdx, ArrayList<Integer> endNodeIdx, boolean setVisited){
		this.points = pts;
		this.connections = connections;
		this.graph = getGraph();
		this.pruningLength = 0.0;
		this.vessels = extractCenterlineFromGraphEndNodes(startNodeIdx, endNodeIdx, setVisited);
	}
	
	private Graph getGraph() {
		// create graph
		Graph costGraph = new Graph();
		
		for(int i = 0; i < points.size(); i++){
			Point sourcePoint = new Point(i,0,0);
			Node sourceNode = new Node(sourcePoint);
			sourceNode.setPhysical(points.get(i));
			costGraph.addNodeNoChecking(sourceNode);	
		}
			
		for(int i = 0; i < connections.size(); i++){
			int[] c = connections.get(i);
			int sourceNodeIdx = c[0];
			int targetNodeIdx = c[1];
			double length = points.get(sourceNodeIdx).euclideanDistance(points.get(targetNodeIdx));
			Node srcN = new Node(sourceNodeIdx,0,0);
			srcN.setPhysical(points.get(sourceNodeIdx));
			Node tgtN = new Node(targetNodeIdx,0,0);
			tgtN.setPhysical(points.get(targetNodeIdx));
			Edge edgeSrc = new Edge(srcN, sourceNodeIdx, tgtN,targetNodeIdx, 1.0f, length);
			costGraph.setEdgeToNode(sourceNodeIdx, edgeSrc);
			Edge edgeTgt = new Edge(tgtN,targetNodeIdx, srcN, sourceNodeIdx, 1.0f, length);
			costGraph.setEdgeToNode(targetNodeIdx, edgeTgt);
		}
		
	
		return costGraph;
	}
		
	/**
	 * Essentially a Dijkstra's algorithm with backtracking.
	 * @return
	 */
	private VesselTree extractCenterlineFromGraph(int startNodeIdx) {
		Node root = null;// = new Node(startPoint);
		
		// extract nodes from graph
		ArrayList<Node> graphNodes = graph.getNodes();
		
		// create new priority queue with descending cost order
		ArrayList<Node> endNodes = new ArrayList<Node>();
		ArrayList<Boolean> used = new ArrayList<Boolean>();
		
		ArrayList<Integer> numCons = getNumberOfConnectionsPerNode(connections);
		int maxCons = 0;
		int nodeIdx = 0;
		for(int i = 0; i < numCons.size(); i++){
			int cons = numCons.get(i);
			if(cons > maxCons){
				maxCons = cons;
				nodeIdx = i;
			}
			if(cons == 1){
				endNodes.add(graphNodes.get(i));
				used.add(false);
			}
		}
		if(startNodeIdx < 0){
			root = graphNodes.get(nodeIdx);
		}else{
			root = graphNodes.get(startNodeIdx);
		}
		
		// create new priority queue with ascending cost order
		PriorityQueue<Node> nodeQueue = new PriorityQueue<Node>();
		// set complete cost of root node to zero
		root.setCompleteCost(0);
		// put root node in queue
		nodeQueue.add(root);

		// treat all nodes in queue
		while(!nodeQueue.isEmpty()) {

			// get the head of the queue
			Node n1 = nodeQueue.poll();
			// extract adjacent edges from current node
			ArrayList<Edge> adjacentEdges = n1.getEdges();

			for (int i = 0; i < adjacentEdges.size(); i++) {
				Edge e = adjacentEdges.get(i);

				Node n2 = graphNodes.get(e.getIdxN2());

				// compute the new complete cost to the target node with cost of current node plus the edge cost
				double edgeCost = e.getCost();
				double costViaCurEdge = n1.getCompleteCost() + edgeCost;
				
				// check if new cost is below the already existing cost at target node
				if (costViaCurEdge <= n2.getCompleteCost()) {
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
		}
		// perform backtracking on graph
		VesselTree tree = new VesselTree();
		
		// TODO do this iteratively!
		
		
		for(int p = 0; p < endNodes.size(); p++){
			// get end node that produces the longest branch
			double maxLength = 0;
			int idx = 0;
			for(int i = 0; i < endNodes.size(); i++){
				if(!used.get(i)){
					double completeLength = 0;
					Node cur = endNodes.get(i);
					Node previous = cur.getPreviousNode();			
					while (previous != null && !previous.isVisited()) {
						completeLength += cur.getEdgeToPrevious().getLength();
						cur = previous;
						previous = previous.getPreviousNode();
					}
					if(completeLength > maxLength){
						maxLength = completeLength;
						idx = i;
					}
				}
			}		
			used.set(idx, true);
			
			double completeLength = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = endNodes.get(idx);

			double[] physCoordCur = cur.getPhysical().getCoordinates();
			branch.add(new VesselBranchPoint(cur.x, cur.y, cur.z,
					physCoordCur[0], physCoordCur[1], physCoordCur[2], 1));
			Node previous = cur.getPreviousNode();
			
			while (previous != null && !previous.isVisited()) {

				completeLength += cur.getEdgeToPrevious().getLength();

				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}

				double[] physicalCoordinates = previous.getPhysical().getCoordinates();
				
				branch.add(new VesselBranchPoint(previous.x, previous.y, previous.z,
						physicalCoordinates[0], physicalCoordinates[1], physicalCoordinates[2], meanRadius));

				previous.setVisited();

				cur = previous;
				previous = previous.getPreviousNode();
			}
			// add last connections
			if(previous != null && previous.isVisited()){
				completeLength += cur.getEdgeToPrevious().getLength();
				
				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}
				double[] physicalCoordinates = previous.getPhysical().getCoordinates();
				
				branch.add(new VesselBranchPoint(previous.x, previous.y, previous.z,
						physicalCoordinates[0], physicalCoordinates[1], physicalCoordinates[2], meanRadius));
			}
			
			branch.setLength(completeLength);
			

			if (completeLength > pruningLength) {
				tree.add(branch);
			}
		}
		return tree;
	}

	
	private VesselTree extractCenterlineFromGraphEndNodes(int startNodeIdx, ArrayList<Integer> endNodeIdcs, boolean setUsed) {
		// extract nodes from graph
		ArrayList<Node> graphNodes = graph.getNodes();
		
		// create new priority queue with descending cost order
		ArrayList<Node> endNodes = new ArrayList<Node>();
		//ArrayList<Boolean> used = new ArrayList<Boolean>();
		
		Node root = graphNodes.get(startNodeIdx);
		for(int i = 0; i < endNodeIdcs.size(); i++){
			endNodes.add(graphNodes.get(endNodeIdcs.get(i)));
		}
		
		// create new priority queue with ascending cost order
		PriorityQueue<Node> nodeQueue = new PriorityQueue<Node>();
		// set complete cost of root node to zero
		root.setCompleteCost(0);
		// put root node in queue
		nodeQueue.add(root);

		// treat all nodes in queue
		while(!nodeQueue.isEmpty()) {

			// get the head of the queue
			Node n1 = nodeQueue.poll();
			// extract adjacent edges from current node
			ArrayList<Edge> adjacentEdges = n1.getEdges();

			for (int i = 0; i < adjacentEdges.size(); i++) {
				Edge e = adjacentEdges.get(i);

				Node n2 = graphNodes.get(e.getIdxN2());

				// compute the new complete cost to the target node with cost of current node plus the edge cost
				double edgeCost = e.getCost();
				double costViaCurEdge = n1.getCompleteCost() + edgeCost;
				
				// check if new cost is below the already existing cost at target node
				if (costViaCurEdge <= n2.getCompleteCost()) {
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
		}
		VesselTree tree = new VesselTree();
		for(int p = 0; p < endNodes.size(); p++){
			double completeLength = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = graphNodes.get(endNodeIdcs.get(p));

			double[] physCoordCur = cur.getPhysical().getCoordinates();
			branch.add(new VesselBranchPoint(cur.x, cur.y, cur.z,
					physCoordCur[0], physCoordCur[1], physCoordCur[2], 1));
			Node previous = cur.getPreviousNode();
			
			while (previous != null && !previous.isVisited()) {

				completeLength += cur.getEdgeToPrevious().getLength();

				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}

				double[] physicalCoordinates = previous.getPhysical().getCoordinates();
				
				branch.add(new VesselBranchPoint(previous.x, previous.y, previous.z,
						physicalCoordinates[0], physicalCoordinates[1], physicalCoordinates[2], meanRadius));
				if(setUsed){
					previous.setVisited();
				}

				cur = previous;
				previous = previous.getPreviousNode();
			}
			// add last connections
			if(previous != null && previous.isVisited()){
				completeLength += cur.getEdgeToPrevious().getLength();
				
				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}
				double[] physicalCoordinates = previous.getPhysical().getCoordinates();
				
				branch.add(new VesselBranchPoint(previous.x, previous.y, previous.z,
						physicalCoordinates[0], physicalCoordinates[1], physicalCoordinates[2], meanRadius));
			}
			
			branch.setLength(completeLength);
			

			if (completeLength > pruningLength) {
				tree.add(branch);
			}
		}
		return tree;
	}

	/**
	 * counts number of connections impinging / emerging from a certain point
	 * @param connections
	 * @return
	 */
	private ArrayList<Integer> getNumberOfConnectionsPerNode(ArrayList<int[]> connections){
		ArrayList<Integer> numberOfConnections = new ArrayList<Integer>();
		for(int i = 0; i < points.size(); i++){numberOfConnections.add(0);}
		for(int i = 0; i < connections.size(); i++){
			int[] con = connections.get(i);
			for(int j = 0; j < 2; j++){
				int val = numberOfConnections.get(con[j]);
				numberOfConnections.set(con[j], val+1);
			}
		}
		return numberOfConnections;
	}

	public ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> getVesselTreeAsEdgeList(){
		ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> list = 
				new ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>();
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			for(int j = 0; j < vb.size()-1; j++){
				PointND bp1 = new PointND(vb.get(j).getPhysCoordinates());
				PointND bp2 = new PointND(vb.get(j+1).getPhysCoordinates());
				list.add(new edu.stanford.rsl.conrad.geometry.shapes.simple.Edge(bp1,bp2));
			}
		}
		return list;
	}
	
	public ArrayList<ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>> getVesselTreeAsEdgeListSep(){
		ArrayList<ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>> list = 
				new ArrayList<ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>>();
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge> l = 
					new ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>();
			for(int j = 0; j < vb.size()-1; j++){
				PointND bp1 = new PointND(vb.get(j).getPhysCoordinates());
				PointND bp2 = new PointND(vb.get(j+1).getPhysCoordinates());
				l.add(new edu.stanford.rsl.conrad.geometry.shapes.simple.Edge(bp1,bp2));
			}
			list.add(l);
		}
		return list;
	}
	
	public ArrayList<ArrayList<Double>> getVesselTreeRadiiAsListSep(){
		ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			ArrayList<Double> l = new ArrayList<Double>();
			for(int j = 0; j < vb.size()-1; j++){
				double br1 = vb.get(j).getRadius();
				double br2 = vb.get(j+1).getRadius();
				l.add((br1+br2)/2);
			}
			list.add(l);
		}
		return list;
	}
	
	public double getPruningLength() {
		return pruningLength;
	}

	public void setPruningLength(double pruningLength) {
		this.pruningLength = pruningLength;
	}

}

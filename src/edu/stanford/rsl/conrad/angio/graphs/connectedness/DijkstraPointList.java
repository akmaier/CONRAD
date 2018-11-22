/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Edge;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Graph;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Node;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranch;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;

public class DijkstraPointList {
	private Graph graph = null;
	private VesselTree vessels = null;
	
	private ArrayList<PointND> points = null;
	
	private double pruningLength = 20;
		
	private PointND startPoint = null;
	private double costThreshold = 0;
	
	private boolean verbose = false;
	
	public void run(ArrayList<PointND> pts, PointND sp, double cth){
		this.points = pts;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.graph = getGraph();
			
		this.vessels = extractCenterlineFromGraph();
	}
	
	public void runOrientationDependent(ArrayList<PointND> pts, PointND sp, double cth){
		this.points = pts;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.graph = getGraphOrientationDependent();
			
		this.vessels = extractCenterlineFromGraph();
	}
		
	public void runSamePoints(PointND sp){
		this.startPoint = sp;			
		this.vessels = extractCenterlineFromGraph();
	}
	
	/**
	 * Compute an undirected graph with all nodes which edges are below the cost threshold.
	 * @param grid 
	 * @param startPoint - start coordinates for tree should lie inside desired vessel tree
	 * @param medThreshold - medialness threshold
	 * @return graph
	 */
	private Graph getGraph() {
		if(verbose)
			System.out.println("Extracting graph.");

		// create graph
		Graph costGraph = new Graph();
		
		for(int i = 0; i < points.size(); i++){
			Point sourcePoint = new Point(i,0,0);
			Node sourceNode = new Node(sourcePoint);
			sourceNode.setPhysical(points.get(i));
			costGraph.addNodeNoChecking(sourceNode);	
		}
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
				
		for(int count = 0; count < points.size(); count++){
			final int i = count;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {
						for(int j = 0; j < points.size(); j++){				
							int sourceNodeIdx = i;
							double cost = points.get(i).euclideanDistance(points.get(j));
							if(cost < costThreshold){
								int targetNodeIdx = j;
								Node srcN = new Node(i,j,0);
								srcN.setPhysical(points.get(i));
								Node tgtN = new Node(i,j,1);
								tgtN.setPhysical(points.get(j));
								Edge edge = new Edge(srcN, sourceNodeIdx, tgtN,targetNodeIdx, (float)cost, 1);
								costGraph.setEdgeToNode(sourceNodeIdx, edge);
							}
						}
					}
				})
			);
		}
		for (Future<?> future : futures){
		   try{
		       future.get();
		   }catch (InterruptedException e){
		       throw new RuntimeException(e);
		   }catch (ExecutionException e){
		       throw new RuntimeException(e);
		   }
		}
		return costGraph;
	}
	
	private Graph getGraphOrientationDependent() {
		if(verbose)
			System.out.println("Extracting graph.");

		// create graph
		Graph costGraph = new Graph();
		
		for(int i = 0; i < points.size(); i++){
			Point sourcePoint = new Point(i,0,0);
			Node sourceNode = new Node(sourcePoint);
			sourceNode.setPhysical(points.get(i));
			costGraph.addNodeNoChecking(sourceNode);	
		}
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
				
		for(int count = 0; count < points.size(); count++){
			final int i = count;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {
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
							if(cost < costThreshold){
								SimpleVector v = new SimpleVector(points.get(j).getAbstractVector());
								v.subtract(points.get(i).getAbstractVector());
								v.normalizeL2();
								int oct = getOctant(v);
								octants.get(oct).add(j);		 
							}
						}
						for(int k = 0; k < octants.size(); k++){
							int minEdgeIdx = 0;
							double minCost = Double.MAX_VALUE;
							for(int j = 0; j < octants.get(k).size(); j++){
								int targetNodeIdx = octants.get(k).get(j);
								double cost = points.get(i).euclideanDistance(points.get(targetNodeIdx));
								if(cost < minCost){
									minCost = cost;
									minEdgeIdx = targetNodeIdx;
								}
							}
							Node srcN = new Node(i,minEdgeIdx,0);
							srcN.setPhysical(points.get(i));
							Node tgtN = new Node(i,minEdgeIdx,1);
							tgtN.setPhysical(points.get(minEdgeIdx));
							Edge edge = new Edge(srcN, i, tgtN,minEdgeIdx, (float)minCost, 1);
							costGraph.setEdgeToNode(i, edge);
							
						}						
					}
				})
			);
		}
		for (Future<?> future : futures){
		   try{
		       future.get();
		   }catch (InterruptedException e){
		       throw new RuntimeException(e);
		   }catch (ExecutionException e){
		       throw new RuntimeException(e);
		   }
		}
		return costGraph;
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
	
	/**
	 * Essentially a Dijkstra's algorithm with backtracking.
	 * @return
	 */
	private VesselTree extractCenterlineFromGraph() {
		Node root = null;// = new Node(startPoint);
		
		// extract nodes from graph
		ArrayList<Node> graphNodes = graph.getNodes();
		
		double minDist = Double.MAX_VALUE;
		for(int i = 0; i < graphNodes.size(); i++){
			Node n = graphNodes.get(i);
			double dist = n.getPhysical().euclideanDistance(startPoint);
			dist = Math.sqrt(dist);
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
			endNodes.add(n1);
			
		}

		// perform backtracking on graph
		if(verbose)
			System.out.println("Backtracking");
		
		VesselTree tree = new VesselTree();
		
		while (!endNodes.isEmpty()) {
			double completeLength = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = endNodes.poll();

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

			branch.setLength(completeLength);

			if (completeLength > pruningLength) {
				tree.add(branch);
			}

		}

		return tree;
	}
	
	public VesselTree getVesselTree(){
		return this.vessels;
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
	
	public int getVesselTreeSize(){
		int count = 0;
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			for(int j = 0; j < vb.size(); j++){
				count++;
			}
		}
		return count;
	}
	
	public int getMaxVesselLength(){
		int length = 0;
		for(int i = 0; i < vessels.size(); i++){
			VesselBranch vb = vessels.get(i);
			length = Math.max(length,vb.size());
		}
		return length;
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
	
	public double getPruningLength() {
		return pruningLength;
	}

	public void setPruningLength(double pruningLength) {
		this.pruningLength = pruningLength;
	}

	public boolean isVerbose() {
		return verbose;
	}

	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
}

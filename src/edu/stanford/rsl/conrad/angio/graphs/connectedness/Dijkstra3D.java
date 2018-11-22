package edu.stanford.rsl.conrad.angio.graphs.connectedness;

import ij.IJ;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Edge;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Graph;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.Node;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranch;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselBranchPoint;
import edu.stanford.rsl.conrad.angio.graphs.connectedness.components.VesselTree;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;
//import edu.stanford.rsl.conrad.visualization.EdgeViewer;

public class Dijkstra3D {
	
	public enum TwentySixConnectedLattice{
		
		// same height: z = 0		
		p1p0p0 (new int[]{1,0,0}),
		p1p1p0 (new int[]{1,1,0}),
		p0p1p0 (new int[]{0,1,0}),
		m1p1p0 (new int[]{-1,1,0}),
		m1m0p0 (new int[]{-1,0,0}),
		m1m1p0 (new int[]{-1,-1,0}),
		m0m1p0 (new int[]{0,-1,0}),
		p1m1p0 (new int[]{1,-1,0}),
		// slice above: z = +1
		p0p0p1 (new int[]{0,0,1}),
		p1p0p1 (new int[]{1,0,1}),
		p1p1p1 (new int[]{1,1,1}),
		p0p1p1 (new int[]{0,1,1}),
		m1p1p1 (new int[]{-1,1,1}),
		m1m0p1 (new int[]{-1,0,1}),
		m1m1p1 (new int[]{-1,-1,1}),
		m0m1p1 (new int[]{0,-1,1}),
		p1m1p1 (new int[]{1,-1,1}),
		// slice above: z = +1
		p0p0m1 (new int[]{0,0,-1}),
		p1p0m1 (new int[]{1,0,-1}),
		p1p1m1 (new int[]{1,1,-1}),
		p0p1m1 (new int[]{0,1,-1}),
		m1p1m1 (new int[]{-1,1,-1}),
		m1m0m1 (new int[]{-1,0,-1}),
		m1m1m1 (new int[]{-1,-1,-1}),
		m0m1m1 (new int[]{0,-1,-1}),
		p1m1m1 (new int[]{1,-1,-1});
		
		private int[] offs = new int[]{0,0};
		
		TwentySixConnectedLattice(int[] offs){
			this.offs = offs;
		}
		
		public int[] getShift(){
			return this.offs;
		}		
	}
	
	private Graph graph = null;
	private VesselTree vessels = null;
	
	private double pruningLength = 20;
		
	private int[] boxStart = null;
	private int[] boxSize = null;
	
	private Grid3D grid = null;
	private Grid3D radii = null;
	private Point startPoint = null;
	private double costThreshold = 0;
	private double endNodeThreshold = 0;
	
	ArrayList<Point> endPoints = null;
	
	
	public static void main(String[] args){
		
		String dir = ".../";
		String file = "magnitudeImg.tif";
		Grid3D g = ImageUtil.wrapImagePlus(IJ.openImage(dir+file));
		
		RoiManager manager = new RoiManager();
		manager.runCommand("Open", dir+"RoiSet.zip");
		Roi[] rois = manager.getRoisAsArray();
		Point startPoint = new Point(rois[1].getBounds().x,rois[1].getBounds().y,rois[1].getPosition()-1);
		ArrayList<Point> endPts = new ArrayList<Point>();
		for(int i = 2; i < rois.length; i++){
			endPts.add(new Point(rois[i].getBounds().x,rois[i].getBounds().y,rois[i].getPosition()-1));
		}
		int[] boxStart = new int[]{rois[0].getBounds().x,rois[0].getBounds().y,0};
		int[] boxSize = new int[]{rois[0].getBounds().width,rois[0].getBounds().height,g.getSize()[2]};
		
		manager.close();
		
		
		Dijkstra3D dijk = new Dijkstra3D();
		dijk.setPruningLength(75);
		dijk.setBoundingBox(boxStart, boxSize);
		dijk.run(g, new Grid3D(g), startPoint, 4.5, endPts);
		ArrayList<PointND> centLin = dijk.getVesselTreeAsList();
		
		MinimumSpanningTree mst = new MinimumSpanningTree(centLin, 2.0);	//2.0
		mst.run();
		/*
		ArrayList<ArrayList<edu.stanford.rsl.conrad.geometry.shapes.simple.Edge>> edges = 
				mst.getMstHierarchical(0);
		*/
		// TODO Wait for migration of visualization.EdgeViewer
		// EdgeViewer.renderEdgesComponents(edges);
	}
	
	/**
	 * Using this method assumes that the complete cost map has already been calculated and is represented by g.
	 * @param g
	 * @param radii
	 * @param sp
	 * @param cth
	 */
	public void run(Grid3D g, Grid3D radii, Point sp, double cth){
		this.grid = g;
		this.radii = radii;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.endNodeThreshold = cth;
		this.graph = getGraph();
			
		this.vessels = extractCenterlineFromGraph();
	}
	
	public void runSamePoints(Point sp){
		this.startPoint = sp;
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
	public void run(Grid3D g, Grid3D radii, Point sp, double cth, double endNodeTh){
		this.grid = g;
		this.radii = radii;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.endNodeThreshold = endNodeTh;
		this.graph = getGraph();
		
		this.vessels = extractCenterlineFromGraph();
	}
	
	public void run(Grid3D g, Grid3D radii, Point sp, double cth, ArrayList<Point> endPts){
		this.grid = g;
		this.radii = radii;
		this.startPoint = sp;
		this.costThreshold = cth;
		this.endNodeThreshold = cth;
		this.endPoints = endPts;
		this.graph = getGraph();
		
		this.vessels = extractCenterlineFromGraphWithEndnodes();
	}
		
	public VesselTree getVesselTree(){
		return this.vessels;
	}
	
	public void setVesselTree(VesselTree vt){
		this.vessels = vt;
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
			double dist = Math.pow(n.x-startPoint.x,2) + Math.pow(n.y-startPoint.y, 2) + Math.pow(n.z-startPoint.z,2);
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
		//System.out.println("Dijkstra computation");

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
		//System.out.println("Backtracking");
		
		VesselTree tree = new VesselTree();
		
		while (!endNodes.isEmpty()) {
			double completeLength = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = endNodes.poll();

			double[] physCoordCur = grid.indexToPhysical(cur.x, cur.y, cur.z);
			branch.add(new VesselBranchPoint(cur.x, cur.y, cur.z,
					physCoordCur[0], physCoordCur[1], physCoordCur[2], cur.getEdgeToPrevious().getRadius()));
			Node previous = cur.getPreviousNode();
			
			while (previous != null && !previous.isVisited()) {

				completeLength += cur.getEdgeToPrevious().getLength();

				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}

				double[] physicalCoordinates = grid.indexToPhysical(previous.x, previous.y, previous.z);

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
	
	/**
	 * Essentially a Dijkstra's algorithm with backtracking.
	 * @return
	 */
	private VesselTree extractCenterlineFromGraphWithEndnodes() {
Node root = null;// = new Node(startPoint);
		
		// extract nodes from graph
		ArrayList<Node> graphNodes = graph.getNodes();
		
		double minDist = Double.MAX_VALUE;
		for(int i = 0; i < graphNodes.size(); i++){
			Node n = graphNodes.get(i);
			double dist = Math.pow(n.x-startPoint.x,2) + Math.pow(n.y-startPoint.y, 2) + Math.pow(n.z-startPoint.z,2);
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
		//System.out.println("Dijkstra computation");

		// treat all nodes in queue
		while (!nodeQueue.isEmpty()) {

			// get the head of the queue
			Node n1 = nodeQueue.poll();
			// extract adjacent edges from current node
			ArrayList<Edge> adjacentEdges = n1.getEdges();

			// treat every edge at the current node
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
			if (isInEndNodes(n1)) {
				endNodes.add(n1);
			}
		}

		// perform backtracking on graph
		//System.out.println("Backtracking");
		
		VesselTree tree = new VesselTree();
		
		while (!endNodes.isEmpty()) {
			double completeLength = 0;
			VesselBranch branch = new VesselBranch();

			Node cur = endNodes.poll();

			double[] physCoordCur = grid.indexToPhysical(cur.x, cur.y, cur.z);
			branch.add(new VesselBranchPoint(cur.x, cur.y, cur.z,
					physCoordCur[0], physCoordCur[1], physCoordCur[2], cur.getEdgeToPrevious().getRadius()));
			Node previous = cur.getPreviousNode();
			
			while (previous != null && !previous.isVisited()) {

				completeLength += cur.getEdgeToPrevious().getLength();

				double meanRadius = 1;
				if (cur.getEdgeToPrevious() == null) {
					meanRadius = cur.getEdgeToPrevious().getRadius();
				}

				double[] physicalCoordinates = grid.indexToPhysical(previous.x, previous.y, previous.z);

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
	
	private boolean isInEndNodes(Node n){
		for(int i = 0; i < endPoints.size(); i++){
			Point p = endPoints.get(i);
			if(p.x-n.x == 0){
				if(p.y-n.y == 0){
					if(p.z-n.z == 0){
						return true;
					}
				}
			}
		}
		return false;
	}
	
	public void setBoundingBox(int[] boxStart, int[] boxSize){
		this.boxStart = boxStart;
		this.boxSize = boxSize;
	}
	
	/**
	 * Compute an undirected graph with all nodes which edges are below the cost threshold.
	 * @param grid 
	 * @param startPoint - start coordinates for tree should lie inside desired vessel tree
	 * @param medThreshold - medialness threshold
	 * @return graph
	 */
	private Graph getGraph() {
		System.out.println("Extracting graph.");

		// create graph
		Graph costGraph = new Graph();
		
		if(boxStart == null || boxSize == null){
			boxStart = new int[]{0,0,0};
			boxSize = new int[]{grid.getSize()[0],grid.getSize()[1],grid.getSize()[2]};
		}
		
		for(int i = boxStart[0]; i < boxStart[0]+boxSize[0]; i++){
			for(int j = boxStart[1]; j < boxStart[1]+boxSize[1]; j++){
				for(int k = boxStart[2]; k < boxStart[2]+boxSize[2]; k++){
					Point sourcePoint = new Point(i,j,k);
					Node sourceNode = new Node(sourcePoint);
					costGraph.addNodeNoChecking(sourceNode);
				}
			}	
		}
		
		if(Configuration.getGlobalConfiguration() == null){
			Configuration.loadConfiguration();
		}
		ExecutorService executorService = Executors.newFixedThreadPool(
				Integer.valueOf(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.MAX_THREADS)));
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
				
		for(int count = boxStart[0]+1; count < boxStart[0]+boxSize[0]-1; count++){
			final int i = count;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {
						int idxx = i-boxStart[0];
						for(int j = boxStart[1]+1; j < boxStart[1]+boxSize[1]-1; j++){				
							int idxy = j-boxStart[1];
							for(int k = boxStart[2]+1; k < boxStart[2]+boxSize[2]-1; k++){
								int idxz = k-boxStart[2];
								int sourceNodeIdx = idxx*boxSize[1]*boxSize[2] + idxy*boxSize[2] + idxz;
								float cost = grid.getAtIndex(i, j, k);
								if(cost < costThreshold){
									for(TwentySixConnectedLattice shift : TwentySixConnectedLattice.values()){
										int[] s = shift.getShift();
										int targetNodeIdx = (idxx+s[0])*boxSize[1]*boxSize[2] + (idxy+s[1])*boxSize[2] + (idxz+s[2]);
										Edge edge = new Edge(new Node(i,j,k), sourceNodeIdx, new Node(i+s[0],j+s[1],k+s[2]),targetNodeIdx, cost, radii.getAtIndex(i, j, k));
										costGraph.setEdgeToNode(sourceNodeIdx, edge);
									}
								}
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

	
	public Grid3D visualizeVesselTree(VesselTree v){
		int[] gSize = this.grid.getSize();
		Grid3D g = new Grid3D(gSize[0], gSize[1], gSize[2]);
		g.setSpacing(this.grid.getSpacing());
		g.setOrigin(this.grid.getOrigin());
		for(int i = 0; i < v.size(); i++){
			VesselBranch b = v.get(i);
			for(int j = 0; j < b.size(); j++){
				Point p = b.get(j);
				g.setAtIndex(p.x, p.y, p.z, 1);
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
	
	public Grid3D visualizeGraphNodes(Graph graph){
		int[] gSize = this.grid.getSize();
		Grid3D g = new Grid3D(gSize[0], gSize[1], gSize[2]);
		g.setSpacing(this.grid.getSpacing());
		g.setOrigin(this.grid.getOrigin());
		ArrayList<Node> nodes = graph.getNodes();
		for(int i = 0; i < nodes.size(); i++){
			Node n = nodes.get(i);
			g.setAtIndex(n.x, n.y, n.z, 1);
		}		
		return g;
	}

	public double getPruningLength() {
		return pruningLength;
	}

	public void setPruningLength(double pruningLength) {
		this.pruningLength = pruningLength;
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
}

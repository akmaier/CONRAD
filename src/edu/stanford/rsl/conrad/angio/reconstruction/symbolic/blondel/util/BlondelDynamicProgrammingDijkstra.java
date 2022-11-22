/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util;

import java.util.ArrayList;
import java.util.PriorityQueue;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;

public class BlondelDynamicProgrammingDijkstra {
	Graph graph = null;
	// weighting constant for the energyterm
	double alpha = 25;

	ArrayList<PointND> optimized3DPoints = null;
	ArrayList<DynamicCorrespondence> possibleCorrespondences;
	Skeleton skelet = null;

	public BlondelDynamicProgrammingDijkstra(ArrayList<DynamicCorrespondence> possibleCorrespondences, Skeleton skel, double alph) {
		this.possibleCorrespondences = possibleCorrespondences;
		this.skelet = skel;
		this.alpha = alph;
		this.optimized3DPoints = new ArrayList<PointND>();
	}

	public Graph getGraph() {
		return this.graph;
	}

	public void run() {

		ArrayList<DynamicCorrespondence> corr = new ArrayList<DynamicCorrespondence>();
		corr.addAll(possibleCorrespondences);
		Skeleton skeletohn = new Skeleton();
		skeletohn.addAll(skelet);
		int pointer = 0;
		// treat every branch on its own create a graph, find the shortest path,
		// add the optimal points
		for (int i = 0; i < skeletohn.size(); i++) {
			// get all points of one branch
			ArrayList<DynamicCorrespondence> corrInSkel = new ArrayList<DynamicCorrespondence>();
			for (int j = 0; j < skeletohn.get(i).size(); j++) {
				corrInSkel.add(corr.get(pointer));
				pointer++;
			}
			// initialize Graph for one branch
			Graph graphInSkel = initializeGraph(corrInSkel);
			// get the shortest path

			ArrayList<PointND> forDummies = optimize(graphInSkel);
			// ArrayList<PointND> forDummies = easyway(graphInSkel, corrInSkel);
			// add the points of the shortest path
			if (forDummies != null) {
				optimized3DPoints.addAll(forDummies);
			}
		}
	}

	public ArrayList<PointND> getOptimized3DPoints() {
		return optimized3DPoints;
	}

	public void setOptimized3DPoints(ArrayList<PointND> optimized3dPoints) {
		optimized3DPoints = optimized3dPoints;
	}

	public Graph initializeGraph(ArrayList<DynamicCorrespondence> corr) {
		Graph output = new Graph();
		// initialize the rootnodes as point 0,0,0 with the corresponding
		// 3DPoint attached and add it to graph with the hash (0,0
		for (int k = 0; k < corr.get(0).getPoints3D().size(); k++) {
			Node n0 = new Node(0, k, 0);
			n0.setPhysical(corr.get(0).getPoints3D().get(0));
			output.addNode("0," + k, n0);
		}
		// checkfirst();
		// run through all points of view 1
		for (int counter = 0; counter < corr.size() - 1; counter++) {

			// get the PointSet for the next view
			ArrayList<PointND> points3DnextCorr = corr.get(counter + 1).getPoints3D();
			// while (points3DnextCorr.size() == 0) {
			// counter++;
			// points3DnextCorr = corr.get(counter + 1).getPoints3D();
			// }
			// get the connectivities between current point and potential next
			// points
			double[][] connectivities = corr.get(counter).getConnectivity();
			// get the errorvalues for each 3DPoint
			double[] errors = corr.get(counter).getErrorList();
			// get the 3DPointset
			ArrayList<PointND> points3D = corr.get(counter).getPoints3D();

			// run through all possible Matches in the next view
			for (int corrNextView = 0; corrNextView < points3DnextCorr.size(); corrNextView++) {
				// Node n1 = this.graph.getNodeAt(new
				// int[]{counter,correspondenceNr}.toString());
				// create a new Node for the next point in view 1 and set its
				// 3DPoint
				Node n2 = new Node(counter + 1, corrNextView, 0);
				n2.setPhysical(new PointND(0, 0, 0));
				n2.setPhysical(corr.get(counter + 1).getPoints3D().get(corrNextView));
				for (int corrView = 0; corrView < points3D.size(); corrView++) {
					// Node n2 = this.graph.getNodeAt(new
					// int[]{counter+1,correspondencesNextViewNr}.toString());
					// Node n2 = this.graph.getNodeAt(new
					// String(counter+1+","+correspondencesNextViewNr));
					// look into the graph and chose the already existing node
					// n1
					Node n1 = output.getNodeAt(counter + "," + corrView);
					// calculate the energyterm
					double error = 100;
					error = errors[corrView];
					double connectivity = 0;
					connectivity = connectivities[corrView][corrNextView];
					double energyterm = error + alpha * connectivity;
					// dont let it be negative
					if (energyterm < 0) {
						energyterm = 0;
					}
					Edge e1 = new Edge(n1, n2, energyterm, counter + 1);
					// Edge e1 = new Edge(n1, n2, energyterm);
					output.getNodes().get(counter + "," + corrView).addEdge(e1);
					// graph.setEdgeToNode(counter, e1);
					n2.addEdge(e1);
					output.addEdge(e1);
				}
				output.addNode(counter + 1 + "," + corrNextView, n2);
			}
		}
		output.setDepth(corr.size());

		return output;

	}

	
	/**
	 * Essentially a Dijkstra's algorithm with backtracking.
	 * 
	 * @return
	 */
	public ArrayList<PointND> optimize(Graph graf) {

		// extract nodes from graph
		// HashMap<String, Node> graphNodes = graf.getNodes();
		ArrayList<PointND> output = new ArrayList<PointND>();

		ArrayList<Node> roots = new ArrayList<Node>();
		double minPath = Double.MAX_VALUE;
		boolean allRoots = false;
		int q = 0;
		while (allRoots == false) {
			if (graf.getNodeAt(("0," + q)) != null) {
				roots.add(graf.getNodeAt(("0," + q)));
				q++;
			} else {
				allRoots = true;
			}
		}
		boolean allEnds = false;
		int w = 0;

		for (int rootNr = 0; rootNr < roots.size(); rootNr++) {
			Node root = roots.get(rootNr);
			ArrayList<Node> nodes = new ArrayList<Node>();

			// create new priority queue with ascending cost order
			PriorityQueue<Node> nodeQueue = new PriorityQueue<Node>();
			// set complete cost of root node to zero
			root.setCompleteCost(0);
			// put root node in queue
			nodeQueue.add(root);

			// Dijkstra computation of the graph
			// System.out.println("Dijkstra computation");
			ArrayList<Node> ends = new ArrayList<Node>();
			while (allEnds == false) {

				if (graf.getNodeAt(("0," + w)) != null) {
					ends.add(graf.getNodeAt((graf.getDepth() + "," + w)));
					w++;
				} else {
					allEnds = true;
				}
			}
			// treat all nodes in queue
			double endCost = Double.MAX_VALUE;
			while (!nodeQueue.isEmpty()) {

				// get the head of the queue
				Node n1 = nodeQueue.poll();
				// extract adjacent edges from current node
				ArrayList<Edge> adjacentEdges = n1.getEdges();

				for (int i = 0; i < adjacentEdges.size(); i++) {
					Edge e = adjacentEdges.get(i);

					Node n2 = (e.getN2());

					// compute the new complete cost to the target node with
					// cost of
					// current node plus the edge cost
					double edgeCost = e.getEnergy();
					double costViaCurEdge = n1.getCompleteCost() + edgeCost;
					endCost = costViaCurEdge;
					// check if new cost is below the already existing cost
					// at
					// target node
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
					nodes.add(n2);
				}
			}
			for (int endNr = 0; endNr < ends.size(); endNr++) {
				if (nodes.size() - endNr - 1 >= 0) {
					Node endNode = nodes.get(nodes.size() - endNr - 1);
					endCost = endNode.getCompleteCost();
					ArrayList<PointND> buffer = new ArrayList<PointND>();
					if (endCost < minPath) {
						while (endNode != root) {

							buffer.add(endNode.getPhysical());
							endNode = endNode.getPreviousNode();
						}
						output = buffer;
						minPath = endCost;
					}
				}
			}
		}
		return output;

	}
}

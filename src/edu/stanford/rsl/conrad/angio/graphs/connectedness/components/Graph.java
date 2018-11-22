/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness.components;

import java.util.ArrayList;

public class Graph {

	/**
	 * Arraylists to saving nodes and edges of the graph.
	 */
	private ArrayList <Edge> edges = null;
	private ArrayList <Node> nodes = null;
	
	private Node root = null;
	
	/**
	 * Create empty graph.
	 */
	public Graph() {
		this.edges = new ArrayList<Edge>();
		this.nodes = new ArrayList<Node>();
	}
	
	/**
	 * Create graph with root node
	 * @param root - root node
	 */
	public Graph(Node root) {
		this.edges = new ArrayList<Edge>();
		this.nodes = new ArrayList<Node>();
		this.setRoot(root);
	}
	
	/**
	 * Set root node.
	 * @param root
	 */
	public void setRoot(Node root) {
		this.root = root;
	}
	
	/**
	 * Get root node.
	 * @return root
	 */
	public Node getRoot() {
		return root;
	}
	
	/**
	 * Add an edge to the graph.
	 * @param e
	 */
	public void addEdge(Edge e) {
		if (this.edges.contains(e)) {
			// do nothing
		} else {
			e.getN1().addEdge(e);
			e.getN2().addEdge(e);
			this.edges.add(e);
		}
	}
	
	/**
	 * add a node to the graph
	 * @param n
	 */
	public void addNode(Node n) {
		if (this.nodes.contains(n)) {
			// do nothing
		} else {
			this.nodes.add(n);
		}
	}
	
	/**
	 * get all edges in the graph
	 * @return
	 */
	public ArrayList<Edge> getEdges() {
		return edges;
	}
	
	/**
	 * get all nodes in the graph
	 * @return
	 */
	public ArrayList<Node> getNodes() {
		return nodes;
	}

	public void setEdgeToNode(int sourceNodeIdx, Edge edge) {
		Node n = nodes.get(sourceNodeIdx);
		n.addEdge(edge);
		nodes.set(sourceNodeIdx,n);
	}

	public void addNodeNoChecking(Node n) {
		this.nodes.add(n);
	}
	
}

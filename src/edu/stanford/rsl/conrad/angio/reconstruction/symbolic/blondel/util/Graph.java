/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util;

import java.util.ArrayList;
import java.util.HashMap;

public class Graph {

	/**
	 * Arraylists to saving nodes and edges of the graph.
	 */
	private ArrayList <Edge> edges = null;
	private HashMap <String,Node> nodes = null;
	private int depth = 0;
	public int getDepth() {
		return depth;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}


	private Node root = null;
	
	/**
	 * Create empty graph.
	 */
	public Graph() {
		this.edges = new ArrayList<Edge>();
		this.nodes = new HashMap<String,Node>();
	}
	
	/**
	 * Create graph with root node
	 * @param root - root node
	 */
	public Graph(Node root) {
		this.edges = new ArrayList<Edge>();
		this.nodes = new HashMap<String,Node>();
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
			e.getN2().addEdge(e);
			this.edges.add(e);
		}
	}
	
	/**
	 * add a node to the graph
	 * @param n
	 */
	public void addNode(String hash,Node n) {
		if (this.nodes.containsKey(hash)) {
			// do nothing
		} else {
			this.nodes.put(hash,n);
		}
	}
	
	/**
	 * get all edges in the graph
	 * @return
	 */
	public ArrayList<Edge> getEdges() {
		return edges;
	}
	
	public void setEdges(ArrayList<Edge> edges) {
		this.edges = edges;
	}

	public void setNodes(HashMap<String, Node> nodes) {
		this.nodes = nodes;
	}
	
	/**
	 * get all nodes in the graph
	 * @return
	 */
	public HashMap<String,Node> getNodes() {
		return nodes;
	}
	public Node getNodeAt(String i){
		if (!this.nodes.containsKey(i)) {		
			
		}
		return this.nodes.get(i);
	}
	

	public void addNodeNoChecking(String hash, Node n) {
		this.nodes.put(hash,n);
	}
	
}
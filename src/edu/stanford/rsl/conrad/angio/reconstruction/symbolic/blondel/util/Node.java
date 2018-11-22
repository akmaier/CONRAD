package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.Point;

public class Node extends Point implements Comparable<Node>{
	
	// list of adjacent edges
	private ArrayList<Edge> edges = null;
	
	private PointND phys = null;
	private int level = 0;
	private int number = 0;
	public int getLevel() {
		return level;
	}


	public void setLevel(int level) {
		this.level = level;
	}


	public int getNumber() {
		return number;
	}


	public void setNumber(int number) {
		this.number = number;
	}

	// indicator of complete costs for minimal cost path extraction
	private double totalCost = Double.MAX_VALUE;
	// visit status
	private boolean visited = false;
	// previous node for Dijkstra
	private Node previous = null;
	private Edge edgeToPrevious = null;
	
	
	/**
	 * Create node with information about position.
	 * @param x
	 * @param y
	 * @param z
	 */
	public Node(int x, int y, int z) {
		super(x, y, z);
		this.edges = new ArrayList<Edge>();
	}
	
	public void setPhysical(PointND p){
		this.phys = p;
	}
	
	public PointND getPhysical(){
		return this.phys;
	}

	/**
	 * Add an adjacent edge to the vertex.
	 * @param e1 - adjacent edge
	 * @return false if edge is already included, true if edge was added
	 */
	public boolean addEdge(Edge e1) {
		if (edges.contains(e1)) {
			return false;
		} else {
			this.edges.add(e1);
			return true;
		}
	}
	
	/**
	 * Get list of adjacent edges.
	 * @return list of adjacent edges
	 */
	public ArrayList<Edge> getEdges() {
		return this.edges;
	}
	
	/**
	 * Set total costs at this node. Used in Dijkstra
	 * @param totalCost - sum of costs to this point
	 */
	public void setCompleteCost(double totalCost) {
		this.totalCost = totalCost;
	}
	
	/**
	 * Get total cost at this node.
	 * @return
	 */
	public double getCompleteCost() {
		return this.totalCost;
	}
	
	/**
	 * Set this nodes status to visited.
	 */
	public void setVisited() {
		this.visited = true;
	}
	
	/**
	 * Set this nodes status to not visited.
	 */
	public void setUnVisited() {
		this.visited = false;
	}
	
	/**
	 * Get current visiting status of node.
	 * @return status
	 */
	public boolean isVisited() {
		return this.visited;
	}
	
	public void setPreviousNode(Node n) {
		this.previous = n;
	}
	
	public Node getPreviousNode() {
		return this.previous;
	}
	
	public void setEdgeToPrevious(Edge e) {
		this.edgeToPrevious = e;
	}
	
	public Edge getEdgeToPrevious() {
		return this.edgeToPrevious;
	}

	@Override
	/**
	 * 
	 */
	public int compareTo(Node other) {
		return Double.compare(totalCost, other.totalCost);
	}
	
}

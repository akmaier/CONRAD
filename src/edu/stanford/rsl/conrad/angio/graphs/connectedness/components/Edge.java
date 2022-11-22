/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.graphs.connectedness.components;

public class Edge {

	private int idxN1 = 0;
	private int idxN2 = 0;
	// first end of the edge
	private Node n1 = null;
	// second end of the edge
	private Node n2 = null;
	// costs of the edge
	private double cost = 0;
	// length of the edge
	private double length = 1;
	// radius at at center of edge
	private double radius = 0;
	
	/**
	 * Create an edge between two nodes with given costs.
	 * @param n1 - first node
	 * @param n2 - second node
	 * @param cost - costs of the edge
	 */
	public Edge(Node n1, Node n2, double cost) {
		this.n1 = n1;
		this.n2 = n2;
		this.cost = cost;
	}
	
	/**
	 * Create an edge between two nodes with given costs and length.
	 * @param n1 - first node
	 * @param n2 - second node
	 * @param cost - costs of the edge
	 * @param length - physical length of the edge
	 */
	public Edge(Node n1, Node n2, double cost, double length) {
		this.n1 = n1;
		this.n2 = n2;
		this.cost = cost;
		this.length = length;
	}
	
	/**
	 * Create an edge between two nodes with given costs, length and radius.
	 * @param n1 - first node
	 * @param n2 - second node
	 * @param cost - cost of the edge
	 * @param length - physical length of the edge
	 * @param radius - physical radius of the edge
	 */
	public Edge(Node n1, Node n2, double cost, double length, double radius) {
		this.n1 = n1;
		this.n2 = n2;
		this.cost = cost;
		this.length = length;
		this.radius = radius;
	}
	
	public Edge(Node node1, int sourceNodeIdx, Node node2, int targetNodeIdx, float edgeCo, double length) {
		this.n1 = node1;
		this.idxN1 = sourceNodeIdx;
		this.n2 = node2;
		this.idxN2 = targetNodeIdx;
		this.cost = edgeCo;
		this.length = length;
	}

	/**
	 * Get the first node of the edge.
	 * @return first node
	 */
	public Node getN1() {
		return this.n1;
	}
	
	/**
	 * Get the second node of the edge.
	 * @return second node
	 */
	public Node getN2() {
		return this.n2;
	}
	
	/**
	 * Set the cost of the edge.
	 * @param cost
	 */
	public void setCost(double cost) {
		this.cost = cost;
	}
	
	/**
	 * Get the cost of the edge
	 * @return cost of the edge
	 */
	public double getCost() {
		return cost;
	}
	
	/**
	 * Set the length of the edge (necessary for later determination of branch length).
	 * @param length
	 */
	public void setLength(double length) {
		this.length = length;
	}
	
	/**
	 * Get the length of the edge (necessary for later determination of branch length).
	 * @return length of the edge
	 */
	public double getLength() {
		return length;
	}
	
	/**
	 * Set the radius at the center of the Edge.
	 * @param radius
	 */
	public void setRadius(double radius) {
		this.radius = radius;
	}
	
	/**
	 * Get the radius from the Edge.
	 * @return length
	 */
	public double getRadius() {
		return radius;
	}

	public int getIdxN2() {
		return idxN2;
	}
	public int getIdxN1() {
		return idxN1;
	}
}

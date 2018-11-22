package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util;



public class Edge {
	/** "tree" edge classification constant for Depth-first search (DFS) */
	public final static int TREE = 0;
	/** "back" edge classification constant for Depth-first search (DFS) */
	public final static int BACK = 1;
	/** not yet defined edge classification constant for Depth-first search (DFS) */
	public final static int UNDEFINED = -1;
	
	/** DFS classification */
	private int type = Edge.UNDEFINED;
	private Node n1 = null;
	/** vertex at the other extreme of the edge */
	private Node n2 = null;
	private double energy = 0;
	private int targetNidx = 0;
	public Edge(
			Node n1,
			Node n2,
			double energy,
			int trgtNidx)
	{
		this.n1 = n1;
		this.n2 = n2;
		this.energy = energy;
		this.targetNidx = trgtNidx;
	}
	public int getTargetNidx() {
		return targetNidx;
	}
	public void setTargetNidx(int targetNidx) {
		this.targetNidx = targetNidx;
	}
	/**
	 * Set DFS type (BACK or TREE)
	 * @param type DFS classification (BACK or TREE)
	 */
	public void setType(int type)
	{
		this.type = type;
	}
	
	public Node getN1() {
		return n1;
	}
	public void setN1(Node n1) {
		this.n1 = n1;
	}
	/**
	 * Get DFS edge type
	 * @return DFS classification type
	 */
	public int getType()
	{
		return this.type;
	}
	
	/**
	 * Get second vertex.
	 * @return second vertex of the edge
	 */
	public Node getN2()
	{
		return this.n2;
	}
	public void setEnergy(double energy)
	{
		this.energy = energy;
	}

	/**
	 * Get edge length
	 * @return calibrated edge length
	 */
	public double getEnergy()
	{
		return this.energy;
	}
	
}

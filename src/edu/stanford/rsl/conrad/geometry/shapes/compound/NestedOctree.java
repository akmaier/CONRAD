package edu.stanford.rsl.conrad.geometry.shapes.compound;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/**
 * Class to model an LinearOctree which uses either CompoundShapes or LinearOctrees as nodes. If the size of the CompoundShape is greater than MAX_NODE_SIZE the node is transformed to a LinearOctree. 
 * @author akmaier
 *
 */
public class NestedOctree extends LinearOctree {

	/**
	 * 
	 */
	private static final long serialVersionUID = 577772012630565223L;
	private static final int MAX_NODE_SIZE = 50;

	public NestedOctree(PointND center) {
		super(center);
	}

	public NestedOctree(PointND min, PointND max) {
		super(min, max);
	}

	public NestedOctree(PointND min, PointND max, double nextRandom) {
		super(min, max, nextRandom);
	}

	public NestedOctree(NestedOctree no){
		super(no);
	}
	
	private boolean isUnbalanced(){
		int totalSize = this.size();
		if (octant1.size() == totalSize) return true;
		if (octant2.size() == totalSize) return true;
		if (octant3.size() == totalSize) return true;
		if (octant4.size() == totalSize) return true;
		if (octant5.size() == totalSize) return true;
		if (octant6.size() == totalSize) return true;
		if (octant7.size() == totalSize) return true;
		if (octant8.size() == totalSize) return true;
		return false;
	}

	@Override
	protected synchronized void init(){
		if (dirty){
			if (octant1.size() > MAX_NODE_SIZE){
				if (!(octant1 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant1.getMin(), octant1.getMax());
					//System.out.println("New tree size: " + octant1.size());
					tree.addAll(octant1);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant1 = tree;
					}
				}
			}
			if (octant2.size() > MAX_NODE_SIZE){
				if (!(octant2 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant2.getMin(), octant2.getMax());
					tree.addAll(octant2);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant2 = tree;
					}
				}
			}
			if (octant3.size() > MAX_NODE_SIZE){
				if (!(octant3 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant3.getMin(), octant3.getMax());
					tree.addAll(octant3);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant3 = tree;
					}
				}
			}
			if (octant4.size() > MAX_NODE_SIZE){
				if (!(octant4 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant4.getMin(), octant4.getMax());
					tree.addAll(octant4);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant4 = tree;
					}
				}
			}
			if (octant5.size() > MAX_NODE_SIZE){
				if (!(octant5 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant5.getMin(), octant5.getMax());
					tree.addAll(octant5);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant5 = tree;
					}
				}
			}
			if (octant6.size() > MAX_NODE_SIZE){
				if (!(octant6 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant6.getMin(), octant6.getMax());
					tree.addAll(octant6);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant6 = tree;
					}
				}
			}
			if (octant7.size() > MAX_NODE_SIZE){
				if (!(octant7 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant7.getMin(), octant7.getMax());
					tree.addAll(octant7);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant7 = tree;
					}
				}
			}
			if (octant8.size() > MAX_NODE_SIZE){
				if (!(octant8 instanceof NestedOctree)) {
					NestedOctree tree = new NestedOctree(octant8.getMin(), octant8.getMax());
					tree.addAll(octant8);
					if (!tree.isUnbalanced()) {
						tree.init();
						octant8 = tree;
					}
				}
			}
			super.init();
		}
	}

	@Override
	public AbstractShape clone() {
		return new NestedOctree(this);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
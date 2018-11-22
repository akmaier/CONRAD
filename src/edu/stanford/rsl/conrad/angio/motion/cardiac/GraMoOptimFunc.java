/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.motion.cardiac;

import java.util.ArrayList;

public abstract class GraMoOptimFunc {
	
	protected int[] labels;
	
	public abstract void init();

	public abstract int[] getLabels();
	
	public abstract int getNumLabels();
	
	public abstract ArrayList<int[]> getNeighbors();
	
	public abstract ArrayList<Double> getNeighborWeights();
	
	public abstract double computeDataTerm(int idx, int labelP);
	
	public abstract double computeNeighborhoodTerm(int labelP, int labelQ);
	
	public abstract void updateVisualization();
	
	public abstract void updateLabels(int[] labels);
	
	public abstract void updateMotionState();

}

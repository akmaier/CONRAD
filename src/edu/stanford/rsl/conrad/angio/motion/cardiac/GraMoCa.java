package edu.stanford.rsl.conrad.angio.motion.cardiac;

import java.util.ArrayList;
import edu.stanford.rsl.conrad.angio.graphs.cuts.GraphCut;

public class GraMoCa {

	boolean verbose = true;

	float currentFlow = Float.MAX_VALUE;
	
	private int[] labels;
	private int numLabels;
	
	private int maxIter = 20;
	
	private double beta = 0.25; // regularization parameter for error function
	private ArrayList<int[]> neighbors;
	private ArrayList<Double> neighborWeights;
	
	private GraMoOptimFunc costFunction;
	private double[] costAtCurrentLabel;
	private double[] costAtAlpha;
	
		
	
	public GraMoCa(GraMoOptimFunc func){
		this.costFunction = func;		
	}
	
	
	public float run(){
		costFunction.init();
		costFunction.updateVisualization();
		init();
		return optimizeAlphaExpansion();
	}
	
	private float optimizeAlphaExpansion(){		
		float minMaxFlow = Float.MAX_VALUE;
		boolean cont = true;
		int iter = 0;
		while(cont && iter < maxIter){
			float flow = Float.MAX_VALUE;
			for(int alpha = 0; alpha < numLabels; alpha++){
				flow = runAlphaExpansion(alpha);
				if(verbose){
					System.out.println("Loop "+String.valueOf(iter+1)+" - Flow at label "+alpha+" : "+flow);
				}
			}
			costFunction.updateMotionState();
			for(int j = 0; j < costAtCurrentLabel.length; j++){
				float v = computeDataTerm(j, labels[j]);
				costAtCurrentLabel[j] = v;
			}
			iter++;
			if(currentFlow < minMaxFlow){
				minMaxFlow = currentFlow;
				cont = true;
			}else{
				cont = false;
			}
		}
		return minMaxFlow;
	}
	
	/**
	 * Performs one Alpha-Expansion move. 
	 * @param alpha
	 * @return
	 */
	private float runAlphaExpansion(int alpha){
		long startTime = System.nanoTime();
		
		int numNodes = labels.length;
		int numEdges = 0;
		ArrayList<Boolean> equalLabel = new ArrayList<Boolean>();
		for(int i = 0; i < neighbors.size(); i++){
			if((labels[neighbors.get(i)[0]] - labels[neighbors.get(i)[1]]) != 0){
				numNodes += 1;
				numEdges += 2;
				equalLabel.add(false);
			}else{
				numEdges += 1;
				equalLabel.add(true);
			}
		}
		GraphCut gc = new GraphCut(numNodes, numEdges);
		
		// set the terminal weights of the graph
		for(int i = 0; i < labels.length; i++){
//			if(verbose){
//				System.out.println("\t Computing node: "+String.valueOf(i+1)+" of "+String.valueOf(labels.length));
//			}
			gc.setTerminalWeights(	i, 
									calculateAlphaWeight(i, labels[i], alpha),
									calculateNotAlphaWeight(i, labels[i], alpha));
		}
		int auxTerminalIdx = labels.length;
		for(int i = 0; i < neighbors.size(); i++){
			if(!equalLabel.get(i)){
				int[] idx = neighbors.get(i);
				gc.setTerminalWeights(	auxTerminalIdx, 
										Float.MAX_VALUE, 
										calculateNeighborhoodTerm(labels[idx[0]], labels[idx[1]], i));
			}
		}
		
		// set the edge weights of the graph
		// edge weights are set for the auxiliary nodes
		int auxNodeIdx = labels.length;
		for(int i = 0; i < neighbors.size(); i++){
			int[] idx = neighbors.get(i);
			if(equalLabel.get(i)){
				// labels are equal, no auxiliary nodes involved
				gc.setEdgeWeight(	idx[0],
								 	idx[1],	
								 	calculateNeighborhoodTerm(labels[idx[0]], alpha, i));
			}else{
				// calculate position of auxiliary node
				gc.setEdgeWeight(	auxNodeIdx,
									idx[0],	
									calculateNeighborhoodTerm(labels[idx[0]], alpha, i));
				gc.setEdgeWeight(	auxNodeIdx,
									idx[1],	
									calculateNeighborhoodTerm(labels[idx[1]], alpha, i));
				auxNodeIdx++;				
			}
				
		}
		// initiate max-flow / min-cut calculation
		float totalFlow = gc.computeMaximumFlow(false, null);
		// loop through nodes and check the terminal they are connected to
		// if connected to alpha, they keep their old label
		// if connected to not-alpha, they change their label to alpha
		int changedLabels = 0;
		boolean improved = false;
		if(totalFlow < currentFlow){
			currentFlow = totalFlow;
			improved = true;
			for(int i = 0; i < labels.length; i++){
				if(gc.getTerminal(i) == GraphCut.Terminal.NOT_ALPHA){
					labels[i] = alpha;
					costAtCurrentLabel[i] = costAtAlpha[i];
					changedLabels++;
				}
			}
		}
		
		costFunction.updateLabels(labels);
		costFunction.updateVisualization();
		
		long endTime = System.nanoTime();
		if(verbose){
			float duration = (endTime - startTime) / 1000000f;
			System.out.println( "Details for expansion move to "+alpha);
			System.out.println( "Time : "+ duration/1000/60 + "min");
			System.out.println("Labels changed : " +changedLabels);
			System.out.println("Total energy improved : " +improved);
		}
		return totalFlow;
	}
	
	/**
	 * Calculates the terminal weight to the not-alpha terminal.
	 * @param idx
	 * @param label
	 * @param alpha
	 * @return
	 */
	private float calculateNotAlphaWeight(int idx, int label, int alpha){
		if(label == alpha){
			return Float.MAX_VALUE;
		}else{
			return (float)costAtCurrentLabel[idx];
		}
	}
	
	/**
	 * Calculates the terminal weight to the alpha terminal.
	 * @param idx
	 * @param label
	 * @param alpha
	 * @return
	 */
	private float calculateAlphaWeight(int idx, int label, int alpha){
		float val = computeDataTerm(idx, alpha);				
		costAtAlpha[idx] = val;
		return val;
	}
	
	/**
	 * Calculates the regularizing term V, that punishes different labels at neighboring image pairs.
	 * @param labelP
	 * @param labelQ
	 * @param neighborhoodidx
	 * @return
	 */
	private float calculateNeighborhoodTerm(int labelP, int labelQ, int neighborhoodidx){
		double err = beta*neighborWeights.get(neighborhoodidx)*
				costFunction.computeNeighborhoodTerm(labelP,labelQ);
		return (float)err;
	}
	
	
	private void init(){
		System.out.println("Starting initialization...");

		this.labels = costFunction.getLabels().clone();
		this.numLabels = costFunction.getNumLabels();
		this.neighbors = costFunction.getNeighbors();
		this.neighborWeights = costFunction.getNeighborWeights();
		
		costAtCurrentLabel = new double[labels.length];
		costAtAlpha = new double[labels.length];
		
		for(int j = 0; j < costAtCurrentLabel.length; j++){
			float v = computeDataTerm(j, labels[j]);
			costAtCurrentLabel[j] = v;
		}
		
		System.out.println("\t Computing initial weights: done.");		
		System.out.println("Initialized.");
	}
	
	
	private float computeDataTerm(int idx, int label){
		double err = costFunction.computeDataTerm(idx, label);		
		return (float)err; 
	}
	
		
		
	public int getMaxIter() {
		return maxIter;
	}

	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}
		
}

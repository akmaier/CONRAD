package edu.stanford.rsl.conrad.angio.motion.respiratory.graphicalMoCo;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.angio.graphs.cuts.GraphCut;
import edu.stanford.rsl.conrad.angio.motion.OneImagePerCycleGating;
import edu.stanford.rsl.conrad.angio.motion.respiratory.MaxErrorBackProjCL;
import edu.stanford.rsl.conrad.angio.motion.respiratory.graphicalMoCo.tools.Container;
import edu.stanford.rsl.conrad.angio.preprocessing.PreprocessingPipeline;
import edu.stanford.rsl.conrad.angio.reconstruction.autofocus.HistoAutofocus;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;

public class StandardParametrization {
	
	boolean verbose = true;

	float currentFlow = Float.MAX_VALUE;
	
	private int[] volSize = new int[]{256,256,256};
	private float[] volSpace = new float[]{0.5f,0.5f,0.5f};
	
	private ArrayList<float[]> allowedShifts;
	private int numShiftsX = 1;
	private int numShiftsY = 1;
	private int numShiftsZ = 15; // should be uneven to sample 0.0 shift!
	private float magShiftX = 1.5f; // in mm
	private float magShiftY = 1.5f; // in mm
	private float magShiftZ = 1.5f; // in mm

	private int maxIter = 5;
	private double beta = 0.2; // regularization parameter for error function
	private int numNeighbors = 4;
	private ArrayList<int[]> neighbors;
	private ArrayList<Double> neighborWeights;
	
	private int startOffs = 0;
	private int numVariables;
	
	private float cmThreshold = 5;//0.1f; // in mm
	
	private Angiogram ang;
	private int numCardPhases = 25;
	private int[] cardiacPhase;
	private int[] imgsAtPhase;
	private int[] currentLabels;
	private double[] costAtCurrentLabel;
	private double[] costAtAlpha;
	
	
	public static void main(String[] args) {
		DataSets datasets = DataSets.getInstance();
		
		int caseID = 3;		
		DataSet ds = datasets.getCase(caseID);
		String outputDir = ds.getDir()+ "eval/";
		
		PreprocessingPipeline prepPipe = new PreprocessingPipeline(ds, outputDir);
		prepPipe.setWriteOutput(false);
		prepPipe.setRefHeartPhase(-1);
		prepPipe.evaluate();
		Angiogram ang = prepPipe.getPreprocessedAngiogram();
		
		StandardParametrization alpha = new StandardParametrization(ang);
		alpha.run();
	}
	
	public StandardParametrization(Angiogram ang){
		this.ang = ang;
	}
	
	public void run(){
		init();
		optimizeAlphaExpansion();
		float[] sx = new float[currentLabels.length];
		float[] sy = new float[currentLabels.length];
		float[] sz = new float[currentLabels.length];
		for(int i = 0; i < currentLabels.length; i++){
			float[] shifts = allowedShifts.get(currentLabels[i]);
			sx[i] = shifts[0];
			sy[i] = shifts[1];
			sz[i] = shifts[2];
		}
		VisualizationUtil.createPlot("X-shifts over cycle", sx).show();
		VisualizationUtil.createPlot("Y-shifts over cycle", sy).show();
		VisualizationUtil.createPlot("Z-shifts over cycle", sz).show();
	}
	
	
	private void init(){
		System.out.println("Starting initialization...");
		// initialize assignment to cardiac phases
		OneImagePerCycleGating gat = new OneImagePerCycleGating();
		cardiacPhase = gat.assignImagesToPhase(ang.getEcg(), numCardPhases);
		imgsAtPhase = gat.getImagesPerPhase();
		// PAYATTENTION
		// we probably should not optimize for the very first image?
		startOffs = 0;
		numVariables = cardiacPhase.length-startOffs; 
		// initialize shifts
		allowedShifts = new ArrayList<float[]>();
		for(int k = 0; k < numShiftsZ; k++){
			float shiftsZ = (numShiftsZ==1)?0.0f:(magShiftZ*(2f/(numShiftsZ-1)*k-1));
			for(int j = 0; j < numShiftsY; j++){
				float shiftsY = (numShiftsY==1)?0.0f:(magShiftY*(2f/(numShiftsY-1)*j-1));
				for(int i = 0; i < numShiftsX; i++){
					float shiftsX = (numShiftsX==1)?0.0f:(magShiftX*(2f/(numShiftsX-1)*i-1));
					allowedShifts.add(new float[]{shiftsX,shiftsY,shiftsZ});
				}
			}
		}
		// initialize neighbors
		neighbors = new ArrayList<int[]>();
		neighborWeights = new ArrayList<Double>();
		double[] pAngs = new double[ang.getEcg().length];
		for (int i = 0; i < pAngs.length; i++) {
			SimpleVector prAxis = ang.getPMatrices()[i].computePrincipalAxis().normalizedL2();
			pAngs[i] = Math.acos(SimpleOperators.multiplyInnerProd(new SimpleVector(1,0,0), prAxis));
		}
		for(int i = 0; i < ang.getEcg().length; i++){
			if(cardiacPhase[i] == -1){
				continue;
			}else{
				for(int j = 1; j < numNeighbors+1; j++){
					if((i+j < ang.getEcg().length) && cardiacPhase[i+j] != -1){
						neighbors.add(new int[]{i,(i+j)});
						double weight = 1.0 / (Math.abs(pAngs[i]-pAngs[i+j])*180/Math.PI);
						neighborWeights.add(weight);
					}
				}
			}
		}
		// initialize current labels and cost
		System.out.println("\t Computing initial weights: this may take some time.");
		int zeroLabel = (numShiftsZ-1)/2*numShiftsX*numShiftsY + (numShiftsY-1)/2*numShiftsX + (numShiftsX-1)/2;
		currentLabels = new int[ang.getEcg().length];
		Arrays.fill(currentLabels, zeroLabel);
		costAtCurrentLabel = new double[currentLabels.length];
		costAtAlpha = new double[currentLabels.length];
		for(int i = 0; i < numCardPhases; i++){
			System.out.println("\t On "+String.valueOf(i+1)+" of "+String.valueOf(numCardPhases)+".");
			Container backPropData = assembleDataAtPhase(i);
			float v = computeDataTerm(backPropData);
			for(int j = 0; j < costAtCurrentLabel.length; j++){
				if(cardiacPhase[j] == i){
					costAtCurrentLabel[j] = v;
				}
			}
		}
		System.out.println("\t Computing initial weights: done.");		
		System.out.println("Initialized.");
	}
	
		
	private void optimizeAlphaExpansion(){		
		System.out.println("Starting the optimization.");
		float minMaxFlow = Float.MAX_VALUE;
		boolean cont = true;
		int iter = 0;
		while(cont && iter < maxIter){
			float flow = Float.MAX_VALUE;
			for(int alpha = 0; alpha < allowedShifts.size(); alpha++){
				flow = runAlphaExpansion(alpha);
				if(verbose){
					System.out.println("Loop "+String.valueOf(iter+1)+" - Flow at label "+alpha+" : "+flow);
				}
			}
			iter++;
			if(flow < minMaxFlow){
				minMaxFlow = flow;
				cont = true;
			}else{
				cont = false;
			}
		}
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
		Container data = assembleDataTestIndex(idx,alpha);
		float val;
		if(data != null){
			val = computeDataTerm(data);
		}else{
			val = pixelCountToErrorMetric(0);
		}		
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
		float[] shiftP = allowedShifts.get(labelP);
		float[] shiftQ = allowedShifts.get(labelQ);
		double err = 0;
		for (int i = 0; i < shiftP.length; i++) {
			err += Math.pow(shiftP[i]-shiftQ[i], 2);
		}
		err = beta*neighborWeights.get(neighborhoodidx)*Math.sqrt(err);
		return (float)err;
	}
	
	/**
	 * Performs one Alpha-Expansion move. 
	 * @param alpha
	 * @return
	 */
	private float runAlphaExpansion(int alpha){
		long startTime = System.nanoTime();
		
		int numNodes = numVariables;
		int numEdges = 0;
		ArrayList<Boolean> equalLabel = new ArrayList<Boolean>();
		for(int i = 0; i < neighbors.size(); i++){
			if(Math.abs(currentLabels[neighbors.get(i)[0]] - currentLabels[neighbors.get(i)[1]]) > 1){
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
		for(int i = 0; i < numVariables; i++){
			if(verbose){
				System.out.println("\t Computing node: "+String.valueOf(i+1)+" of "+String.valueOf(numVariables));
			}
			gc.setTerminalWeights(	i, 
									calculateAlphaWeight(i, currentLabels[i], alpha),
									calculateNotAlphaWeight(i, currentLabels[i], alpha));
		}
		int auxTerminalIdx = numVariables;
		for(int i = 0; i < neighbors.size(); i++){
			if(!equalLabel.get(i)){
				int[] idx = neighbors.get(i);
				gc.setTerminalWeights(	auxTerminalIdx, 
										Float.MAX_VALUE, 
										calculateNeighborhoodTerm(currentLabels[idx[0]], currentLabels[idx[1]], i));
			}
		}
		
		// set the edge weights of the graph
		// edge weights are set for the auxiliary nodes
		int auxNodeIdx = numVariables;
		for(int i = 0; i < neighbors.size(); i++){
			int[] idx = neighbors.get(i);
			if(equalLabel.get(i)){
				// labels are equal, no auxiliary nodes involved
				gc.setEdgeWeight(	idx[0],
								 	idx[1],	
								 	calculateNeighborhoodTerm(currentLabels[idx[0]], alpha, i));
			}else{
				// calculate position of auxiliary node
				gc.setEdgeWeight(	auxNodeIdx,
									idx[0],	
									calculateNeighborhoodTerm(currentLabels[idx[0]], alpha, i));
				gc.setEdgeWeight(	auxNodeIdx,
									idx[1],	
									calculateNeighborhoodTerm(currentLabels[idx[1]], alpha, i));
				auxNodeIdx++;				
			}
				
		}
		// initiate max-flow / min-cut calculation
		float totalFlow = gc.computeMaximumFlow(false, null);
		// loop through nodes and check the terminal they are connected to
		// if connected to alpha, they keep their old label
		// if connected to not-alpha, they change their label to alpha
		int changedLabels = 0;
		if(totalFlow < currentFlow){
			currentFlow = totalFlow;
			for(int i = 0; i < numVariables; i++){
				if(gc.getTerminal(i) == GraphCut.Terminal.NOT_ALPHA){
					currentLabels[i] = alpha;
					costAtCurrentLabel[i] = costAtAlpha[i];
					changedLabels++;
				}
			}
		}
		
		long endTime = System.nanoTime();
		if(verbose){
			float duration = (endTime - startTime) / 1000000f;
			System.out.println( "Calculation time for expansion step with terminal "+alpha+" : "
								+ duration+"ms = " + duration/1000/60 + "min");
			System.out.println("Number of labels changed to "+ alpha +" : "+changedLabels);
		}
		return totalFlow;
	}
	
		
	private Container assembleDataTestIndex(int idx, int alpha) {
		int phase = cardiacPhase[idx];
		if(phase < 0){
			return null;
		}
		Grid3D projs = new Grid3D(ang.getProjections().getSize()[0],
								  ang.getProjections().getSize()[1],
								  imgsAtPhase[phase]);
		Projection[] pms = new Projection[imgsAtPhase[phase]];
		ArrayList<float[]> sh = new ArrayList<float[]>();
		int count = 0;
		for(int k = 0; k < cardiacPhase.length; k++){
			if(cardiacPhase[k] == phase){
				projs.setSubGrid(count, ang.getReconCostMap().getSubGrid(k));
				pms[count] = ang.getPMatrices()[k];
				if(k == idx){
					sh.add(allowedShifts.get(alpha));
				}else{
					sh.add(allowedShifts.get(currentLabels[k]));
				}
				
				count++;
			}
		}
		return new Container(projs,pms,sh);
	}
	
	private float computeDataTerm(Container data){
		float val = 0;
		if(data.pMats != null && data.pMats.length > 3){
			MaxErrorBackProjCL errorBackProj = new MaxErrorBackProjCL(volSize, volSpace, 
					data.pMats, data.img, data.shifts);
			Grid3D cm = errorBackProj.backprojectCL();
			errorBackProj.unload();
			val = HistoAutofocus.evaluateAutoFocus(cm, cmThreshold);
//			new ImageJ();
//			cm.show();
//			System.out.println();
		}
		return pixelCountToErrorMetric(val); 
	}
	
	private float pixelCountToErrorMetric(float count){
		return 1000f/(count+1f); // to make sure there's no division by zero
	}
	
	private Container assembleDataAtPhase(int i) {
		Grid3D projs = new Grid3D(ang.getProjections().getSize()[0],ang.getProjections().getSize()[1],imgsAtPhase[i]);
		Projection[] pms = new Projection[imgsAtPhase[i]];
		ArrayList<float[]> sh = new ArrayList<float[]>();
		int count = 0;
		for(int k = 0; k < cardiacPhase.length; k++){
			if(cardiacPhase[k] == i){
				projs.setSubGrid(count, ang.getReconCostMap().getSubGrid(k));
				pms[count] = ang.getPMatrices()[k];
				sh.add(allowedShifts.get(currentLabels[k]));
				count++;
			}
		}
		return new Container(projs,pms,sh);
	}
	
	public void setMotionParameters(int[] samples, float[] ranges){
		this.numShiftsX = samples[0];
		this.numShiftsY = samples[1];
		this.numShiftsZ = samples[2];
		
		this.magShiftX = ranges[0];
		this.magShiftY = ranges[1];
		this.magShiftZ = ranges[2];
	}

	public float getCmThreshold() {
		return cmThreshold;
	}

	public void setCmThreshold(float cmThreshold) {
		this.cmThreshold = cmThreshold;
	}

	public int getNumCardPhases() {
		return numCardPhases;
	}

	public void setNumCardPhases(int numCardPhases) {
		this.numCardPhases = numCardPhases;
	}

	public int getMaxIter() {
		return maxIter;
	}

	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}
	
}


package edu.stanford.rsl.conrad.angio.motion.respiratory.graphicalMoCo;

import ij.IJ;
import ij.gui.Plot;
import ij.gui.PlotWindow;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.graphs.cuts.GraphCut;
import edu.stanford.rsl.conrad.angio.motion.OneImagePerCycleGating;
import edu.stanford.rsl.conrad.angio.motion.respiratory.MaxErrorBackProjCL;
import edu.stanford.rsl.conrad.angio.motion.respiratory.graphicalMoCo.tools.Container;
import edu.stanford.rsl.conrad.angio.reconstruction.autofocus.HistoAutofocus;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.io.EcgIO;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;

public class BSplineParametrization {
	
	boolean verbose = true;

	float currentFlow = Float.MAX_VALUE;
	
	private int[] volSize = new int[]{256,256,256};//new int[]{128,128,128};//
	private float[] volSpace = new float[]{0.5f,0.5f,0.5f};//new float[]{0.75f,0.75f,0.75f};//
	
	private ArrayList<float[]> allowedShifts;
	private int numShiftsX = 1;
	private int numShiftsY = 1;
	private int numShiftsZ = 25; // should be uneven to sample 0.0 shift!
	private float magShiftX = 1.0f; // in mm
	private float magShiftY = 1.0f; // in mm
	private float magShiftZ = 3.5f; // in mm

	// Paramters for Bspline Motion Model
	private int degree = 2;
	private int numControlPoints = 4;
	private int[] currentLabels;
	
	private int maxIter = 20;
	
	private double beta = 0.05; // regularization parameter for error function
	private ArrayList<int[]> neighbors;
	private ArrayList<Double> neighborWeights;
	
	private float cmThreshold = 5;//0.1f; // in mm
	
	private Angiogram ang;
	private int numCardPhases = 3;
	private int[] cardiacPhase;
	private int[] imgsAtPhase;
	
	private ArrayList<float[]> initialDisplacements;
	
	private double[] costAtCurrentLabel;
	private double[] costAtAlpha;
	
	private PlotWindow window;
	
	public static void main(String[] args) {
//		DataSets datasets = DataSets.getInstance();
//		
//		int caseID = 3;		
//		DataSet ds = datasets.getCase(caseID);
//		String outputDir = ds.getDir()+ "eval/";
//		
//		PreprocessingPipeline prepPipe = new PreprocessingPipeline(ds, outputDir);
//		prepPipe.setWriteOutput(false);
//		prepPipe.setRefHeartPhase(-1);
//		prepPipe.evaluate();
//		Angiogram ang = prepPipe.getPreprocessedAngiogram();
		
		String dir = ".../";
		Projection[] pMats = ProjMatIO.readProjMats(dir+"projtable.txt");
		double[] ecg = EcgIO.readEcg(dir+"cardlin.txt");
		Grid3D costMap = ImageUtil.wrapImagePlus(IJ.openImage(dir+"case1.tif"));
				
		Angiogram ang = new Angiogram(costMap, pMats, ecg);
		ang.setReconCostMap(costMap);
		
		BSplineParametrization alpha = new BSplineParametrization(ang);
		alpha.run();
	}
	
	public BSplineParametrization(Angiogram ang){
		this.ang = ang;
	}
	
	public float run(){
		init();
		return optimizeAlphaExpansion();
	}
	
	
	private void init(){
		System.out.println("Starting initialization...");
		// initialize assignment to cardiac phases
		OneImagePerCycleGating gat = new OneImagePerCycleGating();
		cardiacPhase = gat.assignImagesToPhase(ang.getEcg(), numCardPhases);
		imgsAtPhase = gat.getImagesPerPhase();
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
		// initialize initial displacements
		if(initialDisplacements == null){
			initialDisplacements = new ArrayList<float[]>();
			for(int i = 0; i < ang.getNumProjections(); i++){
				initialDisplacements.add(new float[3]);
			}
		}
		// initialize neighbors
		neighbors = new ArrayList<int[]>();
		neighborWeights = new ArrayList<Double>();
		float deltaPrimAng = 200.0f/(numControlPoints-1);
		for(int i = 0; i < numControlPoints-1; i++){
			neighbors.add(new int[]{i,(i+1)});
			double weight = 1.0 / deltaPrimAng;
			neighborWeights.add(weight);
		}
		// initialize current labels and cost
		System.out.println("\t Computing initial weights: this may take some time.");
		int zeroLabel = (numShiftsZ-1)/2*numShiftsX*numShiftsY + (numShiftsY-1)/2*numShiftsX + (numShiftsX-1)/2;
		currentLabels = new int[numControlPoints];
		Arrays.fill(currentLabels, zeroLabel);
		costAtCurrentLabel = new double[currentLabels.length];
		costAtAlpha = new double[currentLabels.length];
		
		BSpline currentSpline = assembleSpline();
		float v = computeDataTerm(currentSpline);
		for(int j = 0; j < costAtCurrentLabel.length; j++){
			costAtCurrentLabel[j] = v;
		}
		
		System.out.println("\t Computing initial weights: done.");		
		System.out.println("Initialized.");
	}
	
		
	private BSpline assembleSpline() {
		ArrayList<PointND> points = new ArrayList<PointND>();
		for(int i = 0; i < numControlPoints; i++){
			float[] shift = allowedShifts.get(currentLabels[i]);
			points.add(new PointND(shift[0],shift[1],shift[2]));
		}
		BSpline spline = splineFromControlPoints(points);
		
		return spline;
	}
	
	private BSpline assembleSpline(int idx, int alpha) {
		ArrayList<PointND> points = new ArrayList<PointND>();
		for(int i = 0; i < numControlPoints; i++){
			if(i == idx){
				float[] shift = allowedShifts.get(alpha);
				points.add(new PointND(shift[0],shift[1],shift[2]));
			}else{
				float[] shift = allowedShifts.get(currentLabels[i]);
				points.add(new PointND(shift[0],shift[1],shift[2]));
			}	
		}
		BSpline spline = splineFromControlPoints(points);
		return spline;
	}
	
	private BSpline splineFromControlPoints(ArrayList<PointND> cntrl){
		int n = cntrl.size();
		double[] parameters = new double[n];
		for (int i = 0; i < n; i++){
			parameters[i] = i / (n - 1.0);
		}		
		int k = n + degree + 1; //number of knots
		double[] uKnots = new double[k];
		// compute the knot vector
		for (int i = 0; i <= degree; i++) {
			uKnots[i] = 0;
			uKnots[k-i-1] = 1;
		}		
		for (int j = 1; j <= n - degree - 1; j++) {
			double sum = 0;
			for (int i = j; i < j + degree; i++){
				sum += parameters[i];
			}
			uKnots[j + degree] =  sum / degree;
		}
		BSpline bsp = new BSpline(cntrl, uKnots);		
		return bsp;
	}
		
	private float optimizeAlphaExpansion(){		
		System.out.println("Starting the optimization.");
		float[] xVals = new float[ang.getNumProjections()];
		float minmax = 0;
		for(int i = 0; i < initialDisplacements.size(); i++){
			float[] d = initialDisplacements.get(i).clone();
			d[0] = Math.max(Math.abs(d[0]+magShiftX),Math.abs(d[0]-magShiftX));
			d[1] = Math.max(Math.abs(d[1]+magShiftY),Math.abs(d[1]-magShiftY));
			d[2] = Math.max(Math.abs(d[2]+magShiftZ),Math.abs(d[2]-magShiftZ));
			for(int j = 0; j < 3; j++){
				minmax = Math.max(minmax,d[j]);
			}
		}
		for(int i = 0; i < xVals.length; i++){
			xVals[i] = i+1;
		}
		
		ArrayList<float[]> finalSh = getShifts();
		Plot plot = createPlot(xVals, minmax, finalSh);
		if(window == null){
			window = plot.show();
		}else{
			window.drawPlot(plot);
		}
		
		float minMaxFlow = Float.MAX_VALUE;
		boolean cont = true;
		int iter = 0;
		while(cont && iter < maxIter){
			float flow = Float.MAX_VALUE;
			for(int alpha = 0; alpha < allowedShifts.size(); alpha++){
				flow = runAlphaExpansion(alpha);
				finalSh = getShifts();
				Plot update = createPlot(xVals, minmax, finalSh);
				window.drawPlot(update);
				if(verbose){
					System.out.println("Loop "+String.valueOf(iter+1)+" - Flow at label "+alpha+" : "+flow);
				}
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
		BSpline spline = assembleSpline(idx, alpha);
		float val = computeDataTerm(spline);				
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
		
		int numNodes = numControlPoints;
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
		for(int i = 0; i < numControlPoints; i++){
			if(verbose){
				System.out.println("\t Computing node: "+String.valueOf(i+1)+" of "+String.valueOf(numControlPoints));
			}
			gc.setTerminalWeights(	i, 
									calculateAlphaWeight(i, currentLabels[i], alpha),
									calculateNotAlphaWeight(i, currentLabels[i], alpha));
		}
		int auxTerminalIdx = numControlPoints;
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
		int auxNodeIdx = numControlPoints;
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
		boolean improved = false;
		if(totalFlow < currentFlow){
			currentFlow = totalFlow;
			improved = true;
			for(int i = 0; i < numControlPoints; i++){
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
			System.out.println( "Details for expansion move to "+alpha);
			System.out.println( "Time : "+ duration/1000/60 + "min");
			System.out.println("Labels changed : " +changedLabels);
			System.out.println("Total energy improved : " +improved);
		}
		return totalFlow;
	}
		
	private float computeDataTerm(BSpline spline){
		ArrayList<float[]> shifts = sampleShiftsFromSpline(spline);
		float[] vals = new float[numCardPhases];
		
		ExecutorService executorService = Executors.newFixedThreadPool(1);
		Collection<Future<?>> futures = new LinkedList<Future<?>>();
		
		for(int count = 0; count < numCardPhases; count++){
			final int i = count;
			futures.add(
				executorService.submit(new Runnable() {
					@Override
					public void run() {	
						Container data = assembleDataAtPhase(i, shifts);
						if(data.pMats != null && data.pMats.length > 3){
							MaxErrorBackProjCL errorBackProj = new MaxErrorBackProjCL(volSize, volSpace, 
									data.pMats, data.img, data.shifts);
							Grid3D cm = errorBackProj.backprojectCL();
							errorBackProj.unload();
							float val = HistoAutofocus.evaluateAutoFocus(cm, cmThreshold);
							vals[i] = pixelCountToErrorMetric(val);
						}
					}
				})
			);					
		}
		for (Future<?> future : futures){
			   try{
			       future.get();
			   }catch (InterruptedException e){
			       throw new RuntimeException(e);
			   }catch (ExecutionException e){
			       throw new RuntimeException(e);
			   }
		}
		float totalVal = 0;
		for(int i = 0; i < numCardPhases; i++){
			totalVal += vals[i];
		}
//		ArrayList<float[]> sh = getShiftsFromSpline(spline);
//		VisualizationUtil.createPlot("Z-prelim",sh.get(2)).show();		
		return (totalVal/numCardPhases); 
	}
	
	private float pixelCountToErrorMetric(float count){
		return 1000f/(count+1f); // to make sure there's no division by zero
	}
	
	private Container assembleDataAtPhase(int i, ArrayList<float[]> shifts) {
		Grid3D projs = new Grid3D(ang.getProjections().getSize()[0],ang.getProjections().getSize()[1],imgsAtPhase[i]);
		Projection[] pms = new Projection[imgsAtPhase[i]];
		ArrayList<float[]> sh = new ArrayList<float[]>();
		int count = 0;
		for(int k = 0; k < cardiacPhase.length; k++){
			if(cardiacPhase[k] == i){
				projs.setSubGrid(count, ang.getReconCostMap().getSubGrid(k));
				pms[count] = ang.getPMatrices()[k];
				sh.add(shifts.get(k));
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
	
	private ArrayList<float[]> getShifts(){
		BSpline spline = assembleSpline();
		return getShiftsFromSpline(spline);
	}
	
	private ArrayList<float[]> sampleShiftsFromSpline(BSpline spline){
		ArrayList<float[]> shifts = new ArrayList<float[]>();
		for(int i = 0; i < ang.getNumProjections(); i++){
			double u = i/(ang.getNumProjections()-1.0);
			float[] initShift = initialDisplacements.get(i);
			PointND p = spline.evaluate(u);
			shifts.add(new float[]{(float)p.get(0)+initShift[0],
								   (float)p.get(1)+initShift[1],
								   (float)p.get(2)+initShift[2]});
		}
		return shifts;
	}
	
	private ArrayList<float[]> getShiftsFromSpline(BSpline spline){
		ArrayList<float[]> shiftComps = new ArrayList<float[]>();
		ArrayList<float[]> shifts = sampleShiftsFromSpline(spline); 
		float[] sx = new float[shifts.size()];
		float[] sy = new float[shifts.size()];
		float[] sz = new float[shifts.size()];
		for(int i = 0; i < shifts.size(); i++){
			float[] shift = shifts.get(i);
			sx[i] = shift[0];
			sy[i] = shift[1];
			sz[i] = shift[2];
		}
		shiftComps.add(sx);
		shiftComps.add(sy);
		shiftComps.add(sz);
		return shiftComps;
	}
	
	public void setInitialDisplacements(ArrayList<float[]> disp){
		this.initialDisplacements = disp;
	}
	
	public ArrayList<float[]> getFinalDispacements(){
		BSpline spline = assembleSpline();
		return sampleShiftsFromSpline(spline);
	}
	
	public void setPlotWindow(PlotWindow wind){
		this.window = wind;
	}
	
	public PlotWindow getPlotWindow(){
		return window;
	}
	
	public void setDegree(int n){
		this.degree = n;
	}
	
	public void setNumberOfControlPoints(int n){
		this.numControlPoints = n;
	}
	
	private Plot createPlot(float[] xVals, float minmax, ArrayList<float[]> sh){
		Plot plot = new Plot("Current displacements", "Projection idx", "Displacement in mm",
			    xVals, sh.get(0));
		plot.setColor(Color.BLUE);
		plot.setLimits(1, xVals.length, -minmax, minmax);
		plot.draw();
		plot.setColor(Color.GREEN);
		plot.addPoints(xVals, sh.get(1), Plot.LINE);
		plot.draw();
		plot.setColor(Color.RED);
		plot.addPoints(xVals, sh.get(2), Plot.LINE);
		plot.draw();
		plot.addLegend("X\nY\nZ");
		return plot;
	}
}


/*
 * Copyright (C) 2010-2014 Andreas Maier,
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.absorption;

import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.utils.UserUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class PathLengthHistogramCreatingAbsorptionModel extends AbsorptionModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2227409836615130529L;
	private boolean configured = false;
	private boolean debug = true;
	private AbsorptionModel internalModel;
	private HashMap<Material,int[]> histogramMap;
	private boolean doesNotFit = false;
	/**
	 * Bin size in [mm]
	 */
	private int binSize = 1;
	/**
	 * number of bins
	 */
	private int numberOfBins = 250;
	
	
	@Override
	public void configure() throws Exception {
		internalModel = null;
		binSize = UserUtil.queryInt("Enter bin size [mm]", binSize);
		numberOfBins = UserUtil.queryInt("Enter number of Bins:", numberOfBins);
		AbsorptionModel [] models = AbsorptionModel.getAvailableAbsorptionModels();
		internalModel = (AbsorptionModel) UserUtil.chooseObject("Select Absorption Model: ", "Absorption Model", models, models[0]);
		internalModel.configure();
		histogramMap = new HashMap<Material, int[]>();
		configured = true;
	}

	@Override
	public boolean isConfigured() {
		return configured;
	}
	
	private synchronized void addMaterial(Material mat){
		if (!histogramMap.containsKey(mat)){
			int [] newHistogram = new int[numberOfBins];
			histogramMap.put(mat, newHistogram);
		}
	}
	
	private void increment(Material mat, double pathLength){
		int [] histo = histogramMap.get(mat);
		if (histo == null) {
			addMaterial(mat);
			histo = histogramMap.get(mat);
		}
		int index = (int) Math.round(pathLength / binSize);
		if (index < numberOfBins){
			increment(histo, index);
		} else {
			if (!doesNotFit){
				System.out.println("PathLengthHistogramCreatingAbsorptionModel: Histogram does not fit " + pathLength + " mm of " + mat);
				doesNotFit = true;
			}
		}
	}
	
	private synchronized void increment(int [] histo, int index){
		histo[index]++;
	}
	

	@Override
	public double evaluateLineIntegral(ArrayList<PhysicalObject> segments) {
		/**
		 * Store all path lengths in the internal histogram.
		 * Note that this part of the code is called in parallel.
		 * Thus, it needs to be thread-safe.
		 */
		HashMap<Material, Double> localMap = XRayDetector.accumulatePathLenghtForEachMaterial(segments);
		for (Material material :localMap.keySet()){
			increment(material, localMap.get(material));
		}
		// use the internal model to create the correct absorption values;
		return internalModel.evaluateLineIntegral(segments);
	}

	@Override
	public void notifyEndOfRendering(){
		if (debug) System.out.println("Rendering End.");
		/**
		 * Rendering has ended. Now, we can show the histogram contents.
		 */
		double [] xValues = new double [numberOfBins];
		for (int i = 0; i < numberOfBins; i++){
			xValues[i]=i*binSize;
		}
		for (Material mat: histogramMap.keySet()){
			int [] histo = histogramMap.get(mat);
			double [] yValues = new double [numberOfBins];
			for (int i = 0; i < numberOfBins; i++){
				yValues[i]=histo[i];
			}	
			VisualizationUtil.createPlot(xValues, yValues, "Path lengths in " + mat, "length [mm]", "count").show();
		}
	}
	
	@Override
	public String toString() {
		if (configured) return "Histgram of " + internalModel.toString(); 
		return "Path length histogram creating model";
	}

	/**
	 * @return the debug
	 */
	public boolean isDebug() {
		return debug;
	}

	/**
	 * @param debug the debug to set
	 */
	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	/**
	 * @return the internalModel
	 */
	public AbsorptionModel getInternalModel() {
		return internalModel;
	}

	/**
	 * @param internalModel the internalModel to set
	 */
	public void setInternalModel(AbsorptionModel internalModel) {
		this.internalModel = internalModel;
	}

	/**
	 * @return the histogramMap
	 */
	public HashMap<Material, int[]> getHistogramMap() {
		return histogramMap;
	}

	/**
	 * @param histogramMap the histogramMap to set
	 */
	public void setHistogramMap(HashMap<Material, int[]> histogramMap) {
		this.histogramMap = histogramMap;
	}

	/**
	 * @return the doesNotFit
	 */
	public boolean isDoesNotFit() {
		return doesNotFit;
	}

	/**
	 * @param doesNotFit the doesNotFit to set
	 */
	public void setDoesNotFit(boolean doesNotFit) {
		this.doesNotFit = doesNotFit;
	}

	/**
	 * @return the binSize
	 */
	public int getBinSize() {
		return binSize;
	}

	/**
	 * @param binSize the binSize to set
	 */
	public void setBinSize(int binSize) {
		this.binSize = binSize;
	}

	/**
	 * @return the numberOfBins
	 */
	public int getNumberOfBins() {
		return numberOfBins;
	}

	/**
	 * @param numberOfBins the numberOfBins to set
	 */
	public void setNumberOfBins(int numberOfBins) {
		this.numberOfBins = numberOfBins;
	}

	/**
	 * @param configured the configured to set
	 */
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}

}

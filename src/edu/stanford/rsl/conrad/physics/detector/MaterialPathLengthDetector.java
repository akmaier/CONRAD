/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * This absorption model records the path lengths of each material and writes it into a MultiChannelGrid
 * @author akmaier
 *
 */
public class MaterialPathLengthDetector extends XRayDetector {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5709955427012216841L;
	private int numberOfMaterials = 4;
	private HashMap<Material, Integer> materialChannelMap;
	private boolean doesNotFit = false;
	private String [] names;
	private boolean debugMode = false;

	@Override
	public void configure() throws Exception {
		numberOfMaterials = UserUtil.queryInt("SelectNumber of Recorded Materials:", numberOfMaterials);
		materialChannelMap = new HashMap<Material, Integer>();
		names = new String[numberOfMaterials];
		configured = true;
	}

	@Override
	/**
	 * The material hash map and the corresponding names are set to the initial values, i.e. empty list and array of strings with length corresponding to number of materials.
	 */
	public void init() {
		if (configured){
			materialChannelMap = new HashMap<Material, Integer>();
			names = new String[numberOfMaterials];
		}
	}

	private synchronized void addMaterial(Material mat){
		if (!materialChannelMap.containsKey(mat)){
			Integer number = materialChannelMap.keySet().size();
			if (number < numberOfMaterials) names[number]=mat.getName();
			materialChannelMap.put(mat, number);
		}
	}

	@Override
	public void writeToDetector(Grid2D grid, int x, int y, ArrayList<PhysicalObject> segments){
		HashMap<Material, Double> localMap = XRayDetector.accumulatePathLenghtForEachMaterial(segments);
		for (Material mat: localMap.keySet()) {
			MultiChannelGrid2D mGrid = (MultiChannelGrid2D) grid;
			Integer channel = materialChannelMap.get(mat);
			if (channel == null){
				addMaterial(mat);
				channel = materialChannelMap.get(mat);
			}
			if (channel < numberOfMaterials) {
				mGrid.putPixelValue(x, y, channel, localMap.get(mat) * mat.getDensity());
			} else {
				if (!doesNotFit ){
					System.out.println("MaterialPathLengthAbsorptionModel: MultiChannelGrid does not fit " + mat);
					doesNotFit = true;
				}
			}
		}
	}


	@Override
	public void notifyEndOfRendering(){
		if (debugMode) {
			System.out.println("Rendering finished. The following materials were recorded:");
			for (Material mat:materialChannelMap.keySet()){
				System.out.println("Channel " + materialChannelMap.get(mat) + ": " + mat);
			}
		}
	}

	@Override
	public Grid2D createDetectorGrid(int width, int height){
		MultiChannelGrid2D grid = new MultiChannelGrid2D(width, height, numberOfMaterials);
		grid.setChannelNames(names);
		return grid;
	}


	@Override
	public String toString() {
		if (configured){
			return "Material Path Length Detector (" + numberOfMaterials + " materials)";
		} 
		return "Material Path Length Detector";
	}

	/**
	 * @return the numberOfMaterials
	 */
	public int getNumberOfMaterials() {
		return numberOfMaterials;
	}

	/**
	 * @param numberOfMaterials the numberOfMaterials to set
	 */
	public void setNumberOfMaterials(int numberOfMaterials) {
		this.numberOfMaterials = numberOfMaterials;
	}

	/**
	 * @return the materialChannelMap
	 */
	public HashMap<Material, Integer> getMaterialChannelMap() {
		return materialChannelMap;
	}

	/**
	 * @param materialChannelMap the materialChannelMap to set
	 */
	public void setMaterialChannelMap(HashMap<Material, Integer> materialChannelMap) {
		this.materialChannelMap = materialChannelMap;
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
	 * @return the names
	 */
	public String[] getNames() {
		return names;
	}

	/**
	 * @param names the names to set
	 */
	public void setNames(String[] names) {
		this.names = names;
	}


}

/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.data.collection;

import ij.gui.Roi;

import java.io.Serializable;
import java.util.HashMap;


public class DataSets implements Serializable {

	private static final long serialVersionUID = 5906032858015780267L;

	private static HashMap<Integer, DataSet> cases = null;

	private static DataSets instance = null;

	private DataSets(){
		cases = new HashMap<Integer, DataSet>();
		init();
	}

	public static DataSets getInstance(){
		if(instance==null){
			instance = new DataSets();
		}
		return instance;
	}

	public HashMap<Integer, DataSet> getCases() {
		return cases;
	}

	public DataSet getCase(Integer number){
		return cases.get(number);
	}

	private void init(){

		PreproSettings preSet = null;
		ReconSettings recoSet = null;
		PostproSettings postSet = null;

		String dir;
		int caseNumber;
		
		/*
		// Case 1: Example Dataset
		dir = ".../";
		caseNumber = 1;
		cases.put(caseNumber, new DataSet(dir,"proj.tif"));
		// modify pre-processing and segmentation parameters
		preSet = PreproSettings.getDefaultSettings();
		//		pss.setBilatWidth(-1);
		preSet.setBilatWidth(5);
		preSet.setBilatSigmaDomain(1.0);
		preSet.setBilatSigmaPhoto(0.2);
		preSet.setEcgFile(dir+"cardLin.txt");
		preSet.setRoi(new Roi(120,120,550,435));
		// Vesselness
		preSet.setHessianScales(new double[]{0.5, 0.8, 1.1, 1.4, 1.7});
		preSet.setVesselPercentile(new double[]{0.99, 0.94});
		// Koller
		preSet.setCenterlinePercentile(new double[]{0.999, 0.9945});
		preSet.setDijkstraMaxDistance(8.0);
		preSet.setConCompDilation(2);
		preSet.setConCompSize(1000);
		cases.get(caseNumber).setPrepSegSet(preSet);
		// modify reconstruction parameters
		recoSet = ReconSettings.getDefaultSettings();
		recoSet.setNumDepthLabels(1200);//1000
		recoSet.setSourceDetectorDistanceCoverage(0.1);//0.4
		recoSet.setLabelCenterOffset(0.58);//0.5
		recoSet.setPmatFile(dir+"/projtable.txt");
		// exhaustive merging settings
		recoSet.setMaxReprojectionError(1.5d);
		recoSet.setSuppressionRadius(1.5d);
		cases.get(caseNumber).setRecoSet(recoSet);
		// modify postprocessing parameters
		postSet = PostproSettings.getDefaultSettings();
		postSet.setNumSignificantBranches(10);//-1);
		cases.get(caseNumber).setPostproSet(postSet);
		*/
	}
}

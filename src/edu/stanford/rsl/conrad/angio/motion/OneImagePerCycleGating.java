/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.motion;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.io.EcgIO;

public class OneImagePerCycleGating {

	private int[] imgsPerPhase;
	
	public static void main(String[] args) {
		int caseID = 3;		
		DataSets datasets = DataSets.getInstance();
		DataSet ds = datasets.getCase(caseID);
		
		double[] ecg = EcgIO.readEcg(ds.getPreproSet().getEcgFile());
		OneImagePerCycleGating gat = new OneImagePerCycleGating();
		int[] phases = gat.assignImagesToPhase(ecg, 100);
		System.out.println();
	}
	
	public int[] assignImagesToPhase(double[] ecg, int numPhases){		
		ArrayList<int[]> cycles = detectCycles(ecg);
		ArrayList<double[]> range = determineCycleRange(ecg, cycles);
		double[] sampleRange = determineMinMaxForSampling(range);
		int[] phase = assignPhases(ecg, cycles, numPhases, sampleRange);
		return phase;
	}
	
	public ArrayList<ArrayList<Integer>> assignImagesToPhaseNew(double[] ecg, int numPhases){		
		ArrayList<int[]> cycles = detectCycles(ecg);
		ArrayList<double[]> range = determineCycleRange(ecg, cycles);
		double[] sampleRange = determineMinMaxForSampling(range);
		ArrayList<ArrayList<Integer>> phase = assignPhasesNew(ecg, cycles, numPhases, sampleRange);
		return phase;
	}
	
	private ArrayList<ArrayList<Integer>> assignPhasesNew(double[] ecg, ArrayList<int[]> cycles, int numPhases, double[] minMax){		
		double phaseStep = (minMax[1]-minMax[0])/(numPhases-1);
		ArrayList<ArrayList<Integer>> phase = new ArrayList<ArrayList<Integer>>();
		
		for(int i = 0; i < numPhases; i++){
			double hp = minMax[0] + i*phaseStep;
			ArrayList<Integer> list = new ArrayList<Integer>();
			for(int j = 0; j < cycles.size(); j++){
				int[] cyc = cycles.get(j);
				double minDelta = Double.MAX_VALUE;
				int minIdx = -1;
				for(int k = cyc[0]; k < cyc[1]; k++){
					double delt = Math.abs(ecg[k] - hp);
					if(delt < minDelta){
						minDelta = delt;
						minIdx = k;
					}
				}
				if(minDelta < 0.05){
					list.add(minIdx);
				}
			}
			phase.add(list);
		}
		return phase;
	}
	
	
	private int[] assignPhases(double[] ecg, ArrayList<int[]> cycles, int numPhases, double[] minMax){		
		double phaseStep = (minMax[1]-minMax[0])/(numPhases-1);
		int[] phase = new int[ecg.length];
		int[] imgsPerPhase = new int[numPhases];
		Arrays.fill(phase, -1);
		boolean[] used = new boolean[ecg.length];
		for(int i = 0; i < numPhases; i++){
			double hp = minMax[0] + i*phaseStep;
			for(int j = 0; j < cycles.size(); j++){
				int[] cyc = cycles.get(j);
				double minDelta = Double.MAX_VALUE;
				int minIdx = -1;
				for(int k = cyc[0]; k < cyc[1]; k++){
					if(!used[k]){
						double delt = Math.abs(ecg[k] - hp);
						if(delt < minDelta){
							minDelta = delt;
							minIdx = k;
						}
					}
				}
				if(minDelta < 0.05){
					phase[minIdx] = i;
					used[minIdx] = true;
					imgsPerPhase[i]++;
				}
			}
		}
		this.imgsPerPhase = imgsPerPhase;
		return phase;
	}
	
	private double[] determineMinMaxForSampling(ArrayList<double[]> range){
		double[] minMax = new double[]{0,1};
		for (int i = 0; i < range.size(); i++) {
			if(range.get(i)[0] > 0.92){
				minMax[0] = Math.max(minMax[0],range.get(i)[1]);
				minMax[1] = Math.min(minMax[1],range.get(i)[2]);
			}
		}
		return minMax;
	}
	
	private ArrayList<double[]> determineCycleRange(double[] ecg, ArrayList<int[]> cycles){
		ArrayList<double[]> range = new ArrayList<double[]>();
		for(int i = 0; i < cycles.size(); i++){
			double min = Double.MAX_VALUE;
			double max = -Double.MAX_VALUE;
			for(int j = cycles.get(i)[0]; j < cycles.get(i)[1]; j++){
				double val = ecg[j];
				min = Math.min(min, val);
				max = Math.max(max, val);
			}
			range.add(new double[]{max-min, min, max});
		}
		return range;
	}
	
	private ArrayList<int[]> detectCycles(double[] ecg){
		ArrayList<int[]> cycles = new ArrayList<int[]>();
		int[] currCyc = new int[2];
		double currentVal = ecg[0];
		for(int i = 1; i < ecg.length; i++){
			if(ecg[i] > currentVal){
				currentVal = ecg[i];
			}else{
				currentVal = ecg[i];
				currCyc[1] = i;
				if(currCyc[1]-currCyc[0] > 3){
					cycles.add(currCyc);
				}
				currCyc = new int[]{i,0};
			}
		}
		currCyc[1] = ecg.length;
		if(currCyc[1]-currCyc[0] > 3){
			cycles.add(currCyc);
		}
		return cycles;
	}
	
	public int[] getImagesPerPhase(){
		return this.imgsPerPhase;
	}
	
}

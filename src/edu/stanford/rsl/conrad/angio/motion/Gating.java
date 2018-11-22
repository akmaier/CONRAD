package edu.stanford.rsl.conrad.angio.motion;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;

public class Gating {
	
	private ArrayList<Integer> indexList;
	
	private double width = 0.4; //windowWith [0,1]
	private double exp = 4.0; // exponentA >= 0
	private double refHeartPhase = 0.9; // refHeartPhase [0,1]

	
	public static void main(String[] args){
		
		String imgfile = ".../proj.tif";
		String ecgfile = ".../linTime.txt";
		
		Grid3D imp = ImageUtil.wrapImagePlus(IJ.openImage(imgfile));
		
		Gating gtng = new Gating(0.4, 4, 0.9);
		
		double[] ecg = gtng.readEcg(ecgfile);
		
		Grid3D gated = gtng.applyGatingWeights(imp, ecg);
		
		new ImageJ();
		gated.show();
		
		
	}
	
	
	public Gating(double width, double exp, double hp){
		assert(width <= 1 && width >= 0) : new IllegalArgumentException("Width needs to be in [0,1].");
		assert(exp >= 1) : new IllegalArgumentException("Exp needs to be >1.");
		assert(hp <= 1 && hp >= 0) : new IllegalArgumentException("Heart Phase needs to be in [0,1].");
		this.width = width;
		this.exp = exp;
		this.refHeartPhase = hp;
	}

	/**
	 * Calculates the ECG-gating based weights of projections corresponding to the ECG signal handled.
	 * @param ecg
	 * @return
	 */
	public double[] evaluate(double[] ecg){
		double[] weights = new double[ecg.length];
		for(int i = 0; i < ecg.length; i++){
			if(ecg[i] >= 0){
				weights[i] = getWeight(ecg[i]);
			}
		}
		return weights;
	}
	
	
	private double getWeight(double currentHeartPhase){
		double weight = 0;
		double distMeasure = getDistanceMeasure(currentHeartPhase);
		if(distMeasure > width/2){
			return 0;
		}else{
			double cos = Math.cos((distMeasure/width)*Math.PI);
			weight = Math.pow(cos, exp);
		}		
		return weight;
	}
	
	
	private double getDistanceMeasure(double currentHeartPhase){

		int[] j = new int[] {-1,0,+1};
		
		double min = Math.abs(currentHeartPhase - refHeartPhase + j[0]);
		min = Math.min((Math.abs(currentHeartPhase - refHeartPhase + j[1])),min);
		min = Math.min((Math.abs(currentHeartPhase - refHeartPhase + j[2])),min);
		
		return min;
	}
	
	public Grid3D weightProjections(Grid3D g, double[] ecg){
		double[] weights = evaluate(ecg);
		Grid3D weighted = new Grid3D(g);
		for(int k = 0; k < g.getSize()[2]; k++){
			float weight = (float)weights[k];
			for(int i = 0; i < g.getSize()[0]; i++){
				for(int j = 0; j < g.getSize()[1]; j++){
					weighted.setAtIndex(i, j, k, g.getAtIndex(i, j, k)*weight);
				}
			}	
		}
		return weighted;
	}
	
	public ImagePlus applyGating(ImagePlus a, double[] ecg, double threshold){
		ImageStack ims = a.getImageStack();
		double[] weights = evaluate(ecg);
		this.indexList = new ArrayList<Integer>();
		for(int i = 0; i < ecg.length; i++){
			if(weights[i] > threshold){
				indexList.add(i);
			}
		}
		ImageStack reducedIms = new ImageStack(ims.getProcessor(1).getWidth(), ims.getProcessor(1).getHeight());
		for(int i = 0; i < weights.length; i++){
			if(weights[i] > threshold){
				reducedIms.addSlice(ims.getProcessor(i+1));
			}
		}
		ImagePlus reducedImp = new ImagePlus();
		reducedImp.setStack(reducedIms);
		return reducedImp;
	}
	
	public Grid3D applyGatingWeights(Grid3D a, double[] ecg){
		double[] weights = evaluate(ecg);
		Grid3D weighted = new Grid3D(a);
		int[] gSize = weighted.getSize();
		for(int k = 0; k < weights.length; k++){
			float weight = (float)weights[k];
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					weighted.setAtIndex(i, j, k, a.getAtIndex(i, j, k)*weight);
				}
			}
		}
		return weighted;
	}
	
	/**
	 * Applies gating to an angiogram using the specified threshold for heart-phase weights.
	 * Angiograms consist of projection matrices, projection images and the corresponding ECG signal. 
	 * @param a
	 * @param threshold
	 * @return
	 */
	public Angiogram applyGating(Angiogram a, double threshold){
		Projection[] pMat = a.getPMatrices();
		Grid3D img = a.getProjections();
		double[] primA = a.getPrimAngles();
		double[] secA = a.getSecondaryAngles();
		if(secA == null){
			secA = new double[pMat.length];
		}
		double[] ecg = a.getEcg();
		
		double[] weights = evaluate(ecg);
		int nAbove = 0;
		for(int i = 0; i < ecg.length; i++){
			if(weights[i] > threshold){
				nAbove++;
			}
		}
		double[] reducedEcg = new double[nAbove];
		double[] reducedPrimA = new double[nAbove];
		double[] reducedSecA = new double[nAbove];
		Grid3D reducedImg = new Grid3D(img.getSize()[0],img.getSize()[1],nAbove);
		Projection[] reducedPMat = new Projection[nAbove];
		int count = 0;
		for(int i = 0; i < weights.length; i++){
			if(weights[i] > threshold){
				reducedEcg[count] = ecg[i];
				reducedImg.setSubGrid(count, img.getSubGrid(i));;
				reducedPrimA[count] = primA[i];
				reducedSecA[count] = secA[i];
				reducedPMat[count] = pMat[i];
				count++;
			}
		}
		reducedImg.setSpacing(img.getSpacing());
		reducedImg.setOrigin(img.getOrigin());
		return new Angiogram(reducedImg, reducedPMat, reducedPrimA, reducedSecA, reducedEcg);
	}
	
	public double[] applyGating(double[] a, double[] ecg, double threshold){
		
		double[] weights = evaluate(ecg);
		int nAbove = 0;
		for(int i = 0; i < ecg.length; i++){
			if(weights[i] > threshold){
				nAbove++;
			}
		}
		double[] reducedA = new double[nAbove];
		int count = 0;
		for(int i = 0; i < weights.length; i++){
			if(weights[i] > threshold){
				reducedA[count] = a[i];
				count++;
			}
		}
		return reducedA;
	}
	
	/**
	 * Applies gating to an angiogram by multiplication of the projection with the corresponding weight.
	 * Angiograms consist of projection matrices, projection images and the corresponding ECG signal. 
	 * @param a
	 * @return
	 */
	public Angiogram applyGating(Angiogram a){
		Projection[] pMat = a.getPMatrices();
		Grid3D ims = (Grid3D)a.getProjections().clone();
		double[] primA = a.getPrimAngles();
		double[] ecg = a.getEcg();
		
		double[] weights = evaluate(ecg);
		for(int k = 0; k < weights.length; k++){
			for(int i = 0; i < ims.getSize()[0]; i++){
				for(int j = 0; j < ims.getSize()[1]; j++){
					ims.multiplyAtIndex(i, j, k, (float)weights[k]);
				}
			}			
		}
		return new Angiogram(ims, pMat, primA, ecg);
	}
	
	/**
	 * Reads the ECG from a file. It is assumed that there are as many heart phases in a separate line
 	 * as there are projections.
	 * @param filename
	 * @return
	 */
	private double[] readEcg(String filename){
		ArrayList<Double> e = new ArrayList<Double>();
		FileReader fr;
		try {
			fr = new FileReader(filename);		
			BufferedReader br = new BufferedReader(fr);			
			String line = br.readLine();
			while(line != null){
				StringTokenizer tok = new StringTokenizer(line);
				e.add(Double.parseDouble(tok.nextToken()));
				line = br.readLine();
			}
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		double[] ecg = new double[e.size()];
		for(int i = 0; i < e.size(); i++){
			ecg[i] = e.get(i);
		}
		return ecg;
	}
	
	public ArrayList<Integer> getSelectedSlices(){
		assert (this.indexList != null) : new Exception("Run gating first.");
		return this.indexList;
	}
	
	
	
	
	
	public static Angiogram applyGating(Angiogram a, double refHP, double width, double slope, double threshold){
		if(refHP < 0){
			return new Angiogram(a);
		}
		
		Projection[] pMat = a.getPMatrices();
		Grid3D img = a.getProjections();
		double[] primA = a.getPrimAngles();
		double[] secA = a.getSecondaryAngles();
		if(secA == null){
			secA = new double[pMat.length];
		}
		double[] ecg = a.getEcg();
		
		double[] weights = evaluateStatic(ecg, refHP, width, slope);
		int nAbove = 0;
		for(int i = 0; i < ecg.length; i++){
			if(weights[i] > threshold){
				nAbove++;
			}
		}
		double[] reducedEcg = new double[nAbove];
		double[] reducedPrimA = new double[nAbove];
		double[] reducedSecA = new double[nAbove];
		Grid3D reducedImg = new Grid3D(img.getSize()[0],img.getSize()[1],nAbove);
		Projection[] reducedPMat = new Projection[nAbove];
		int count = 0;
		for(int i = 0; i < weights.length; i++){
			if(weights[i] > threshold){
				reducedEcg[count] = ecg[i];
				reducedImg.setSubGrid(count, img.getSubGrid(i));
				reducedPrimA[count] = primA[i];
				reducedSecA[count] = secA[i];
				reducedPMat[count] = pMat[i];
				count++;
			}
		}
		reducedImg.setSpacing(img.getSpacing());
		reducedImg.setOrigin(img.getOrigin());
		Angiogram prep = new Angiogram(reducedImg, reducedPMat, reducedPrimA, reducedSecA, reducedEcg);
		if(a.getReconCostMap() != null){
			Grid3D cm = new Grid3D(reducedImg);
			count = 0;
			for(int i = 0; i < weights.length; i++){
				if(weights[i] > threshold){
					cm.setSubGrid(count, a.getReconCostMap().getSubGrid(i));
					count++;
				}
			}
			prep.setReconCostMap(cm);
		}
		
		return prep;
	}
	
	public static int[] applyGating(double[] ecg, double refHP, double width, double slope, double threshold){
		double[] weights = evaluateStatic(ecg, refHP, width, slope);
		int nAbove = 0;
		for(int i = 0; i < ecg.length; i++){
			if(weights[i] > threshold){
				nAbove++;
			}
		}
		int[] indcs = new int[nAbove];
		int count = 0;
		for(int i = 0; i < weights.length; i++){
			if(weights[i] > threshold){
				indcs[count] = i;
				count++;
			}
		}
		return indcs;
	}
	
	/**
	 * Calculates the ECG-gating based weights of projections corresponding to the ECG signal handled.
	 * @param ecg
	 * @param refHeartPhase
	 * @param width
	 * @param slope
	 * @return
	 */
	public static double[] evaluateStatic(double[] ecg, double refHeartPhase, double width, double slope){
		double[] weights = new double[ecg.length];
		for(int i = 0; i < ecg.length; i++){
			double weight = 0;
			if(ecg[i] >= 0){
				double distMeasure = getDistanceMeasureStatic(refHeartPhase, ecg[i]);
				if(distMeasure <= width/2){
					double cos = Math.cos((distMeasure/width)*Math.PI);
					weight = Math.pow(cos, slope);
				}
			}
			weights[i] = weight;
		}
		return weights;
	}
		
	private static double getDistanceMeasureStatic(double refHeartPhase, double currentHeartPhase){
		int[] j = new int[] {-1,0,+1};		
		double min = Math.abs(currentHeartPhase - refHeartPhase + j[0]);
		min = Math.min((Math.abs(currentHeartPhase - refHeartPhase + j[1])),min);
		min = Math.min((Math.abs(currentHeartPhase - refHeartPhase + j[2])),min);		
		return min;
	}
	
}

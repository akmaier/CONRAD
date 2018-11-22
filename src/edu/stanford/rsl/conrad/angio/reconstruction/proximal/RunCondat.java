/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal;

import java.util.ArrayList;

import ij.IJ;
import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.motion.OneImagePerCycleGating;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.linearOperators.SpatialGradient;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.linearOperators.TemporalGradient;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.proximableFunctions.CharacteristicFunctionRplus;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.proximableFunctions.OneTwoNorm;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.smoothFunctions.SquaredResidual;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSet;
import edu.stanford.rsl.conrad.angio.util.data.collection.DataSets;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;
import edu.stanford.rsl.conrad.angio.util.io.EcgIO;
import edu.stanford.rsl.conrad.angio.util.io.ProjMatIO;

public class RunCondat {
	
	public static void main(String[] args) {
		
		boolean reconstructStatic = true;
		
		int caseID = 101;
				
		DataSets datasets = DataSets.getInstance();
		DataSet ds = datasets.getCase(caseID);
		
		Projection[] ps = ProjMatIO.readProjMats(ds.getRecoSet().getPmatFile());
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(ds.getProjectionFile()));
		double[] ecg = (reconstructStatic)?new double[ps.length]:EcgIO.readEcg(ds.getPreproSet().getEcgFile());
		Angiogram ang = new Angiogram(img, ps, ecg);
		
		int[] gSize = new int[]{256,256,256};
		double[] gSpace = new double[]{1.0,1.0,1.0};//new double[]{0.75,0.75,0.75};//
		double[] gOrigin = new double[]{-(gSize[0]-1)/2f*gSpace[0],-(gSize[1]-1)/2f*gSpace[1],-(gSize[2]-1)/2f*gSpace[2]}; 
		
		int numPhases = 5;
		int[] phaseAssignment;
		int[] imgsPerPhase;
		if(reconstructStatic){
			phaseAssignment = new int[ps.length];
			imgsPerPhase = new int[]{ps.length};
		}else{
			OneImagePerCycleGating gat = new OneImagePerCycleGating();
			phaseAssignment = gat.assignImagesToPhase(ang.getEcg(), numPhases);
			imgsPerPhase = gat.getImagesPerPhase();
		}
		
		ArrayList<MultiChannelGrid3D> x = new ArrayList<MultiChannelGrid3D>();
		for(int i = 0; i < imgsPerPhase.length; i++){
			MultiChannelGrid3D multCh = new MultiChannelGrid3D(gSize[0], gSize[1], gSize[2], 1);
			multCh.setSpacing(gSpace);
			multCh.setOrigin(gOrigin);
			x.add(multCh);
		}
				
		float lambdaSpat = (float)(0.000625f * ang.getNumProjections() / Math.pow(ang.getProjections().getSpacing()[0],2)
				* Math.pow(gSpace[0],2) / x.size());
		float lambdaTemp = (float)(0.00125f * ang.getNumProjections() / Math.pow(ang.getProjections().getSpacing()[0],2)
				* Math.pow(gSpace[0],3));
		
		SquaredResidual sqrRes = new SquaredResidual(ang);
		sqrRes.setPhaseInformation(phaseAssignment, imgsPerPhase);
		CharacteristicFunctionRplus charRplus = new CharacteristicFunctionRplus();
		ArrayList<ProximableFunction> profuns = new ArrayList<ProximableFunction>();
		OneTwoNorm otnormSpat = new OneTwoNorm(lambdaSpat);
		profuns.add(otnormSpat);
		if(!reconstructStatic){
			OneTwoNorm otnormTemp = new OneTwoNorm(lambdaTemp);
			profuns.add(otnormTemp);
		}
		ArrayList<LinearOperator> linops = new ArrayList<LinearOperator>();
		SpatialGradient gradSpat = new SpatialGradient();
		linops.add(gradSpat);
		if(!reconstructStatic){
			TemporalGradient gradTemp = new TemporalGradient();
			linops.add(gradTemp);
		}
			
		float beta = (float)Math.sqrt(3)*gSize[0]*ang.getNumProjections()/x.size();
		float lStarL = 0;
		for(int i = 0; i < linops.size(); i++){
			lStarL += linops.get(i).getForwBackOperatorNorm();
		}
		float fractionOfMaxTau = 0.75f;
		float tau = (2.0f/beta)*fractionOfMaxTau;
		float sigma = (0.99f - beta/2.0f*tau) / (lStarL*tau);
		int maxIter = 150;
		
		CondatOptimization condatOpt = new CondatOptimization();
		condatOpt.setSmoothFunction(sqrRes);
		condatOpt.setProximableFunction(charRplus);
		condatOpt.setListProximableFunctions(profuns);
		condatOpt.setListLinearOperators(linops);
		ArrayList<MultiChannelGrid3D> res = condatOpt.optimize(x, sigma, tau, 1.0f, maxIter);
		
		new ImageJ();
		res.get(0).getChannel(0).show();
	}

}

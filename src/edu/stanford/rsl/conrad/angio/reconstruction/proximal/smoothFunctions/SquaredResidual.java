/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal.smoothFunctions;

import ij.ImageJ;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.angio.motion.respiratory.graphicalMoCo.tools.Container;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.SmoothFunction;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.util.BackProjCL;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.util.ForwProjCL;
import edu.stanford.rsl.conrad.angio.util.data.organization.Angiogram;

public class SquaredResidual extends SmoothFunction{

	private Angiogram ang;
	
	private int[] phaseAssignment;
	private int[] imgsPerPhase;
		
	public SquaredResidual(Angiogram ang){
		this.ang = ang;
	}
	
	public void setPhaseInformation(int[] phaseAssignment, int[] imgsPerPhase){
		this.phaseAssignment = phaseAssignment;
		this.imgsPerPhase = imgsPerPhase;
	}
	
	@Override
	public float evaluate(ArrayList<MultiChannelGrid3D> x) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void evaluateGradient(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xGrad) {
		
		if(imgsPerPhase == null){
			imgsPerPhase = new int[]{ang.getNumProjections()};
			phaseAssignment = new int[ang.getNumProjections()];
		}
		
		int[] gSize = x.get(0).getSize();
		double[] gSpace = x.get(0).getSpacing(); 
		double[] gOrigin = x.get(0).getOrigin(); 
		
		for(int i = 0; i < x.size(); i++){
			Container data = assembleDataAtPhase(i);
			ForwProjCL fp = new ForwProjCL();
			fp.configure(x.get(i).getChannel(0), 
					data.img.getSize(), data.img.getSpacing(), data.pMats);
			Grid3D sino = fp.project();
			fp.unload();
			Grid3D diff = (Grid3D)NumericPointwiseOperators.subtractedBy(sino, data.img);
						
			BackProjCL cb = new BackProjCL(gSize,gSpace,gOrigin,data.pMats);
			Grid3D recon = cb.backprojectPixelDrivenCL(diff);
			cb.unload();
			MultiChannelGrid3D reconMult = xGrad.get(i);
			reconMult.setChannel(0, recon);
			xGrad.set(i, reconMult);
		}	
	}
	
	private Container assembleDataAtPhase(int i) {
		Grid3D projs = new Grid3D(ang.getProjections().getSize()[0],ang.getProjections().getSize()[1],imgsPerPhase[i]);
		projs.setSpacing(ang.getProjections().getSpacing());
		projs.setOrigin(ang.getProjections().getOrigin());
		Projection[] pms = new Projection[imgsPerPhase[i]];
		int count = 0;
		for(int k = 0; k < phaseAssignment.length; k++){
			if(phaseAssignment[k] == i){
				projs.setSubGrid(count, ang.getProjections().getSubGrid(k));
				pms[count] = ang.getPMatrices()[k];
				count++;
			}
		}
		return new Container(projs,pms);
	}
	
}

/*
 * Copyright (C) 2010-2018 Mathias Unberath, Oliver Taubmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal;

import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.Concatenator;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.util.CondatTools;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.util.MultiChannelGridOperators;

/**
 * Implements 
 * Condat, Laurent. "A generic proximal algorithm for convex optimization—application 
 * to total variation minimization."
 * IEEE Signal Processing Letters 21.8 (2014): 985-989.
 * @author Oliver & Mathias
 *
 */
public class CondatOptimization {
	
	private SmoothFunction f;
	
	private ProximableFunction g;
	
	private ArrayList<ProximableFunction> hs;
	
	private ArrayList<LinearOperator> ls;

	public static void main(String[] args) {
		MultiChannelGrid3D g1 = new MultiChannelGrid3D(5, 6, 7, 2);
		g1.setSpacing(0.1,0.2,0.3);
		g1.setOrigin(-1,-2,-3);
		g1.putPixelValue(0, 0, 0, 0, 1);
		
		MultiChannelGrid3D g2 = new MultiChannelGrid3D(5, 6, 7, 2);
		g2.setSpacing(0.1,0.2,0.3);
		g2.setOrigin(-1,-2,-3);
		g2.putPixelValue(0, 0, 0, 0, 2);
		
		ArrayList<MultiChannelGrid3D> x = new ArrayList<MultiChannelGrid3D>();
		x.add(g1);
		x.add(g2);
			
		
		CondatOptimization condatOpt = new CondatOptimization();
		condatOpt.optimize(x, 1.0f, 1.0f, 1.0f, 10);
	}
	
	public ArrayList<MultiChannelGrid3D> optimize(ArrayList<MultiChannelGrid3D> x, float sigma, float tau, float rho, int maxIter){
		assert(hs.size() == ls.size()):new IllegalArgumentException(
				"Number of ProximableFunctions needs to be the same as LinearOperators!");
		
		new ImageJ();
		ImagePlus imp = getVisualizationStack(x);
		imp.setTitle("Reconstruction result");
		imp.show();
		imp.updateAndDraw();
		
		int[] gSize = x.get(0).getSize();
		int gChannels = x.get(0).getNumberOfChannels();
		double[] gSpace = x.get(0).getSpacing();
		double[] gOrigin = x.get(0).getOrigin();
		
		CondatTools tools = new CondatTools();
		ArrayList<MultiChannelGrid3D> xTildeNew = tools.newEmptyMultiChannelGridList(x.size(),gSize,gChannels,gSpace,gOrigin);
		ArrayList<MultiChannelGrid3D> xNew = tools.newEmptyMultiChannelGridList(x.size(),gSize,gChannels,gSpace,gOrigin);
		ArrayList<MultiChannelGrid3D> gradient = tools.newEmptyMultiChannelGridList(x.size(),gSize,gChannels,gSpace,gOrigin);
		ArrayList<MultiChannelGrid3D> backtransformed = tools.newEmptyMultiChannelGridList(x.size(),gSize,gChannels,gSpace,gOrigin);
		ArrayList<MultiChannelGrid3D> sumBacktransformed = tools.newEmptyMultiChannelGridList(x.size(),gSize,gChannels,gSpace,gOrigin);
		
		ArrayList<ArrayList<MultiChannelGrid3D>> transformed = new ArrayList<ArrayList<MultiChannelGrid3D>>();
		ArrayList<ArrayList<MultiChannelGrid3D>> transformedNew = new ArrayList<ArrayList<MultiChannelGrid3D>>();
		ArrayList<ArrayList<MultiChannelGrid3D>> transformedTmp = new ArrayList<ArrayList<MultiChannelGrid3D>>();
		for(int i = 0; i < hs.size(); i++){
			transformed.add(tools.newEmptyMultiChannelGridList(x.size(),gSize,ls.get(i).getNumberOfComponents(),gSpace,gOrigin));
			transformedNew.add(tools.newEmptyMultiChannelGridList(x.size(),gSize,ls.get(i).getNumberOfComponents(),gSpace,gOrigin));
			transformedTmp.add(tools.newEmptyMultiChannelGridList(x.size(),gSize,ls.get(i).getNumberOfComponents(),gSpace,gOrigin));
		}
		
		// Real deal starting from here
		// ! Note that we allocated all members as the values are supposed to change inplace !
		for(int iter = 0; iter < maxIter; iter++){
			System.out.println("On iteration "+String.valueOf(iter+1)+" of maximal "+String.valueOf(maxIter)+" iterations.");
			System.out.println("\t Updating primal...");
			f.evaluateGradient(x, gradient);
						
			for(int m = 0; m < hs.size(); m++){
				ls.get(m).applyAdjoint(transformed.get(m), backtransformed);
				MultiChannelGridOperators.apbList(sumBacktransformed, backtransformed);
			}
			MultiChannelGridOperators.apbList(gradient, sumBacktransformed);
			xTildeNew = MultiChannelGridOperators.axpbyList(x, gradient, 1f, -tau);
			g.evaluateProx(xTildeNew, xTildeNew, tau);
			xNew = MultiChannelGridOperators.axpbyList(xTildeNew, x, rho, 1f-rho);
			
			System.out.println("\t Updating dual...");
			xTildeNew = MultiChannelGridOperators.axpbyList(xTildeNew, x, 2f, -1f);
			for(int m = 0; m < hs.size(); m++){
				ls.get(m).apply(xTildeNew, transformedTmp.get(m));
				transformedNew.set(m, 
						MultiChannelGridOperators.axpbyList(transformed.get(m), transformedTmp.get(m), 1f, sigma));
				hs.get(m).evaluateConjugateProx(transformedNew.get(m), transformedTmp.get(m), sigma);
				transformed.set(m,
						MultiChannelGridOperators.axpbyList(transformedTmp.get(m), transformed.get(m), rho, 1f-rho));
			}
			x = xNew;
			
			sumBacktransformed = tools.newEmptyMultiChannelGridList(x.size(),gSize,gChannels,gSpace,gOrigin);
			
			ImagePlus conc = getVisualizationStack(x);
			imp.setStack(conc.getImageStack());
			imp.updateAndDraw();
			
		}
		return x;
	}
	
	
	public void setSmoothFunction(SmoothFunction f){
		this.f = f;
	}
	
	public void setProximableFunction(ProximableFunction g){
		this.g = g;
	}
	
	public void setListLinearOperators(ArrayList<LinearOperator> ls){
		this.ls = ls;
	}
	
	public void setListProximableFunctions(ArrayList<ProximableFunction> hs){
		this.hs = hs;
	}
	
	private ImagePlus getVisualizationStack(ArrayList<MultiChannelGrid3D> list){
		ImagePlus[] imps = new ImagePlus[list.size()];
		for(int i = 0; i < list.size(); i++){
			imps[i] = ImageUtil.wrapGrid3D(list.get(i).getChannel(0), "");
		}
		Concatenator conc = new Concatenator();
		return conc.concatenate(imps, false);
	}
	
}

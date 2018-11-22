/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.background;

import ij.IJ;
import ij.ImagePlus;
import ij.io.Opener;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class SpectralInterpolation extends Inpainting {

	int iterations = 100;
	int windowSize = 64;
	double blankRadius = 6;

	public ArrayList<ArrayList<double[]>> twoDpoints = null;

	public SpectralInterpolation(){
		super();
		parameters = new Number[3];
		parameters[0] = 100.0;
		parameters[1] = 64.0;
		parameters[2] = 6.0;
	}

	public SpectralInterpolation(boolean[][][] mask) {
		super(mask);
		parameters = new Number[3];
		parameters[0] = 100.0;
		parameters[1] = 64.0;
		parameters[2] = 6.0;
	}

	public void setTwoDPoints(ArrayList<ArrayList<double[]>> twoDp){
		this.twoDpoints = twoDp;
	}

	@Override
	public Grid3D applyToGrid(Grid3D input) {
		Grid3D output = new Grid3D(input);
		if(twoDpoints == null){
			for (int i = 0; i < output.getSize()[2]; i++) {
				float[] buffer = new float[mask[0].length*mask[0][0].length];
				for (int j = 0; j < mask[i].length; j++) {
					for (int k = 0; k < mask[i][0].length; k++) {
						buffer[j*mask[i][0].length + k] = (mask[i][j][k]==true) ? 0.f : 1.f; 
					}
				}
				spectralInterpolationWorker(output.getSubGrid(i), 
						new Grid2D(buffer,mask[i][0].length,mask[i].length), 
						//(output.getSize()[0]<512 || output.getSize()[1]<512)
						false
						);
			}
		}
		else{
			ArrayList<Grid2D> gSave = new ArrayList<Grid2D>(twoDpoints.size()*twoDpoints.get(0).size());
			for (int i = 0; i < twoDpoints.size(); i++) {
				for (int j = 0; j < twoDpoints.get(i).size(); j++) {
					double[] p = twoDpoints.get(i).get(j);
					int slice = (int)p[2];
					if (slice < input.getSize()[2] && slice >= 0){

						// extract image + mask
						Grid2D g = new Grid2D(windowSize,windowSize);
						Grid2D w = new Grid2D(windowSize,windowSize);
						int[] spoint = {(int)Math.round(p[0]-windowSize/2.0),(int)Math.round(p[1]-windowSize/2.0)};

						for (int y = 0; y < g.getSize()[1]; y++) {
							for (int x = 0; x < g.getSize()[0]; x++) {
								int xx = spoint[0]+x;
								int yy = spoint[1]+y;
								if (xx < 0)
									xx = 0;
								if (yy < 0)
									yy=0;
								if (xx >= input.getSize()[0])
									xx =  input.getSize()[0]-1;
								if (yy >= input.getSize()[1])
									yy =  input.getSize()[1]-1;
								g.setAtIndex(x, y, input.getAtIndex(xx, yy, slice));
								w.setAtIndex(x, y, mask[slice][yy][xx] ? 0.f : 1.f);
							}
						}

						spectralInterpolationWorker(g, w, true);
						gSave.add(g);
					}
				}
			}
			
			int k = 0;
			for (int i = 0; i < twoDpoints.size(); i++) {
				for (int j = 0; j < twoDpoints.get(i).size(); j++) {
					Grid2D g = gSave.get(k);
					k++;
					double[] p = twoDpoints.get(i).get(j);
					int slice = (int)p[2];
					if (slice < input.getSize()[2] && slice >= 0){
						int[] spoint = {(int)Math.round(p[0]-windowSize/2.0),(int)Math.round(p[1]-windowSize/2.0)};
						for (int y = 0; y < g.getSize()[1]; y++) {
							for (int x = 0; x < g.getSize()[0]; x++) {
								int xx = spoint[0]+x;
								int yy = spoint[1]+y;
								if (xx < 0)
									continue;
								if (yy < 0)
									continue;
								if (xx >= input.getSize()[0])
									continue;
								if (yy >= input.getSize()[1])
									continue;

								if (mask[slice][yy][xx] && (new PointND(p[0],p[1])).euclideanDistance(new PointND(xx,yy)) < blankRadius){
									output.setAtIndex(xx, yy, slice, g.getAtIndex(x, y));
								}
							}
						}
					}
				}
			}
		}
		return output;
	}

	@Override
	public void configure() {
		super.configure();
		if (parameters != null && parameters.length >= 3){
			windowSize = (int) Math.ceil((Double) parameters[1]);
			iterations = (int) Math.ceil((Double) parameters[0]);
			blankRadius = (Double) parameters[2];
		}
		else{
			try {
				windowSize=UserUtil.queryInt("Enter window width and height", 64);
				iterations = UserUtil.queryInt("Enter No Iterations", 100);
				blankRadius = UserUtil.queryDouble("Enter blank radius", 6.0);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}

	@Override
	public void setName() {
		name = "SpectralInterpolation";
		parameterNames = new String[]{"NoIterations","WindowSize"};

	}

	@Override
	public String[] getNames() {
		ArrayList<String> out = null;
		if (name != null && parameterNames != null){
			out = new ArrayList<String>(parameterNames.length+1);
			out.add(name);
			out.addAll(Arrays.asList(parameterNames));
		}
		return out.toArray(new String[out.size()]);
	}

	public void spectralInterpolationWorker(Grid2D g, Grid2D w, boolean zeroPadSignal){

		double maxDeltaE_G_Ratio = Double.POSITIVE_INFINITY;
		double maxDeltaE_G_Ratio_Tres = 1.0e-6;

		Grid2DComplex G = new Grid2DComplex(g,zeroPadSignal);
		G.transformForward();

		Grid2DComplex W = new Grid2DComplex(w,zeroPadSignal);
		W.transformForward();

		int[] dim = G.getSize();
		int[] halfDim = {dim[0]/2, dim[1]/2};

		Grid2DComplex Fhat = new Grid2DComplex(dim[0],dim[1],false);
		Grid2DComplex FhatNext = new Grid2DComplex(dim[0],dim[1],false);

		double lastEredVal=0;
		// the iteration loop
		for (int i = 0; i < iterations; i++) {
			// check convergence criterion
			if (maxDeltaE_G_Ratio <= maxDeltaE_G_Ratio_Tres){
				CONRAD.log(String.format("maxDeltaE_G_Ratio: %.15f", maxDeltaE_G_Ratio));
				break;
			}
			// In the i-th iteration select the line pair s1,t1
			// which maximizes the energy reduction [Paragraph after Eq. (16) in the paper]
			double maxDeltaE_G = Double.NEGATIVE_INFINITY;
			ArrayList<Integer[]> sj1=null;
			for (int j = 0; j < dim[0]; j++) {
				for (int k = 0; k < halfDim[1]+1; k++) {
					double val = G.getAtIndex(j, k);
					if (val > maxDeltaE_G){
						sj1= new ArrayList<Integer[]>();
						sj1.add(new Integer[]{j,k});
						maxDeltaE_G = val;
					}
					else if (val == maxDeltaE_G){
						sj1.add(new Integer[]{j,k});
					}
				}
			}
			int idx = (int) Math.floor(Math.random()*sj1.size());
			int s1 = sj1.get(idx)[0];
			int t1 = sj1.get(idx)[1];

			//System.out.println("s-value: " + s1 +" / " + "t-value: " + t1);

			// Calculate the ratio of energy reduction
			// in comparison to the last iteration
			if (i > 0)
				maxDeltaE_G_Ratio = Math.abs((maxDeltaE_G - lastEredVal)/maxDeltaE_G);

			// Save the last energy reduction value for next iteration
			lastEredVal = maxDeltaE_G;

			// Compute the corresponding linepair s2,t2:
			// mirror the positions at halfDim
			int s2 = (s1 > 0) ? dim[0]-(s1%dim[0]) : s1;
			int t2 = (t1 > 0) ? dim[1]-(t1%dim[1]) : t1;

			//System.out.println("s2-value: " + s2 +" / " + "t2-value: " + t2);

			// Special case (0,0)
			/*
			if (s1==0 && t1==0){
				s2 = -1; 
				t2 = -1;
			}*/

			int twice_s1 = (2*s1) % dim[0];
			int twice_t1 = (2*t1) % dim[1];

			boolean specialCase = false;
			//special case
			if ((s1==0 && t1==0) ||
					(s1==0 && t1==halfDim[1]) ||
					(s1==halfDim[0] && t1==0) ||
					(s1==halfDim[0] && t1==halfDim[1])){

				specialCase = true;
				Complex FhatNextval = new Complex(FhatNext.getRealAtIndex(s1, t1),FhatNext.getImagAtIndex(s1, t1));
				Complex Gval = new Complex(G.getRealAtIndex(s1, t1),G.getImagAtIndex(s1, t1));
				Complex Wval = new Complex(W.getRealAtIndex(0, 0),W.getImagAtIndex(0, 0));
				Complex res = FhatNextval.add(Gval.mul(dim[0]*dim[1]).div(Wval));

				FhatNext.setRealAtIndex(s1, t1, (float)res.getReal());
				FhatNext.setImagAtIndex(s1, t1, (float)res.getImag());
			}
			else{
				Complex FhatNextval_s1t1 = new Complex(FhatNext.getRealAtIndex(s1, t1),FhatNext.getImagAtIndex(s1, t1));
				Complex FhatNextval_s2t2 = new Complex(FhatNext.getRealAtIndex(s2, t2),FhatNext.getImagAtIndex(s2, t2));
				Complex Gval = new Complex(G.getRealAtIndex(s1, t1),G.getImagAtIndex(s1, t1));
				Complex Wval00 = new Complex(W.getRealAtIndex(0, 0),W.getImagAtIndex(0, 0));
				Complex WvalTwice = new Complex(W.getRealAtIndex(twice_s1, twice_t1),W.getImagAtIndex(twice_s1,twice_t1));

				Complex tval = ((Gval.mul(Wval00)).sub((Gval.getConjugate().mul(WvalTwice)))).mul((dim[0]*dim[1]));
				tval = tval.div(Wval00.getMagn()*Wval00.getMagn()-WvalTwice.getMagn()*WvalTwice.getMagn());

				Complex res1 = FhatNextval_s1t1.add(tval);
				Complex res2 = FhatNextval_s2t2.add(tval.getConjugate());

				FhatNext.setRealAtIndex(s1, t1,(float) res1.getReal());
				FhatNext.setImagAtIndex(s1, t1,(float) res1.getImag());
				FhatNext.setRealAtIndex(s2, t2,(float) res2.getReal());
				FhatNext.setImagAtIndex(s2, t2,(float) res2.getImag());
			}

			/*			FhatNext.getRealSubGrid(0, 0, dim[0], dim[1]).show("FhatNext_Real_" + i);
			FhatNext.getImagSubGrid(0, 0, dim[0], dim[1]).show("FhatNext_Imag_" + i);

			G.getRealSubGrid(0, 0, dim[0], dim[1]).show("G_Real_Before_" + i);
			G.getImagSubGrid(0, 0, dim[0], dim[1]).show("G_Imag_Before_" + i);*/

			updateSpectrum(G,FhatNext,Fhat,W,s1,t1,specialCase);

			//G.getRealSubGrid(0, 0, dim[0], dim[1]).show("G_Real_" + i);
			//G.getImagSubGrid(0, 0, dim[0], dim[1]).show("G_Imag_" + i);

			G.setAtIndex(s1, t1, 0);
			if (!specialCase) 
				G.setAtIndex(s2, t2, 0);

			Fhat = new Grid2DComplex(FhatNext);
		}

		Fhat.transformInverse();
		for (int j = 0; j < g.getSize()[1]; j++) {
			for (int i = 0; i < g.getSize()[0]; i++) {
				if(w.getAtIndex(i, j)==0){
					g.setAtIndex(i, j, Fhat.getRealAtIndex(i, j));
				}
			}
		}
	}

	public void updateSpectrum(Grid2DComplex G, Grid2DComplex FhatNext,
			Grid2DComplex Fhat, Grid2DComplex W, int s1, int t1, boolean specialCase) {

		int[] sz = FhatNext.getSize();
		Complex Fst = new Complex(FhatNext.getRealAtIndex(s1, t1)-Fhat.getRealAtIndex(s1, t1),
				FhatNext.getImagAtIndex(s1, t1)-Fhat.getImagAtIndex(s1, t1));
		Complex Fstc = Fst.getConjugate();

		int divNr = sz[0]*sz[1];

		for (int j = 0; j < sz[1]; j++) {
			for (int i = 0; i < sz[0]; i++) {
				if (specialCase){
					int xneg = (i-s1)%sz[0];
					int yneg = (j-t1)%sz[1];
					if (xneg < 0)
						xneg = sz[0]+xneg;
					if (yneg < 0)
						yneg = sz[1]+yneg;
					Complex Wval = new Complex(W.getRealAtIndex(xneg, yneg),W.getImagAtIndex(xneg,yneg));
					Complex Gval = new Complex(G.getRealAtIndex(i, j),G.getImagAtIndex(i, j));
					Gval = Gval.sub((Fst.mul(Wval)).div(divNr));
					G.setRealAtIndex(i, j, (float)Gval.getReal());
					G.setImagAtIndex(i, j, (float)Gval.getImag());
				}
				else{
					int xpos = (i+s1)%sz[0];
					int ypos = (j+t1)%sz[1];
					int xneg = (i-s1)%sz[0];
					int yneg = (j-t1)%sz[1];
					if (xneg < 0)
						xneg = sz[0]+xneg;
					if (yneg < 0)
						yneg = sz[1]+yneg;

					Complex Wpos = new Complex(W.getRealAtIndex(xpos, ypos),W.getImagAtIndex(xpos,ypos));
					Complex Wneg = new Complex(W.getRealAtIndex(xneg, yneg),W.getImagAtIndex(xneg,yneg));
					Complex Gval = new Complex(G.getRealAtIndex(i, j),G.getImagAtIndex(i, j));
					Gval = Gval.sub(((Fst.mul(Wneg)).add(Fstc.mul(Wpos))).div(divNr));
					G.setRealAtIndex(i, j, (float)Gval.getReal());
					G.setImagAtIndex(i, j, (float)Gval.getImag());

				}
			}
		}


	}

}

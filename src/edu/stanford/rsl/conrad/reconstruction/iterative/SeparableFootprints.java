package edu.stanford.rsl.conrad.reconstruction.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public class SeparableFootprints {
	
	protected int iumin, iumax, ivmin, ivmax;
	double[] weightU;
	double[] weightV;
	protected final int MAX_WEIGHT_LENGTH = 6;
	
	public SeparableFootprints(){
	}
	
	public boolean rectFootprintWeightU( double umin, double umax, final int boundary ){
		
		if ( umin > umax ){
			double temp = umax;
			umax = umin;
			umin = temp;
		}
		
		umin = Math.max(umin, 0);
		umax = Math.min(umax, boundary);

		if (umin >= umax){
			return false;
		}

		iumin = (int)umin;
		iumax = (int)umax;
		iumax = Math.min(iumax, boundary-1);

		iumax = Math.min(iumax, iumin+MAX_WEIGHT_LENGTH-1);

		weightU = new double[]{1,1,1,1,1,1};

		if ( iumax == iumin){
			weightU[0] = umax - umin;
		}else{
			weightU[0] = iumin + 1 - umin;
			weightU[iumax - iumin] = umax - iumax;
		};
		return true;
	}
	
	public boolean rectFootprintWeightV( double vmin, double vmax, final int boundary ){
		
		if ( vmin > vmax ){
			double temp = vmax;
			vmax = vmin;
			vmin = temp;
		}
		
		vmin = Math.max(vmin, 0);
		vmax = Math.min(vmax, boundary);

		if (vmin >= vmax){
			return false;
		}

		ivmin = (int)vmin;
		ivmax = (int)vmax;
		ivmax = Math.min(ivmax, boundary-1);

		ivmax = Math.min(ivmax, ivmin+MAX_WEIGHT_LENGTH-1);

		weightV = new double[]{1,1,1,1,1,1};

		if ( ivmax == ivmin){
			weightV[0] = vmax - vmin;
		}else{
			weightV[0] = ivmin + 1 - vmin;
			weightV[ivmax - ivmin] = vmax - ivmax;
		};
		return true;
	}
	
	public void footprintsProject( Grid3D projImage, Grid3D volImage, double amplitude, int i, int j, int k, int ip){
		
		double tempVal = volImage.getAtIndex(i, j, k) * amplitude ;
		
		for ( int iu = iumin, iiu = 0; iu <= iumax; iu++, iiu++ ){
			double temp = tempVal * weightU[iiu];
			for (int iv = ivmin, iiv = 0; iv <= ivmax; iv++, iiv++){
				projImage.addAtIndex(ip, iu, iv, (float)(temp*weightV[iiv]));
			} //iv
		} //iu
		
	}
	
	public void footprintsBackproject( Grid3D projImage, Grid3D volImage, double amplitude, int i, int j, int k, int ip){
		
		double tempVal = 0;
		for ( int iu = iumin, iiu = 0; iu <= iumax; iu++, iiu++ ){
			double sum = 0;
			for (int iv = ivmin, iiv = 0; iv <= ivmax; iv++, iiv++){
				sum += weightV[iiv] * projImage.getAtIndex(ip, iu, iv);
			} //iv
			tempVal += sum * weightU[iiu];
		} //iu

		tempVal = tempVal * amplitude;
		volImage.addAtIndex(i, j, k, (float) tempVal);
	
	}
	

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
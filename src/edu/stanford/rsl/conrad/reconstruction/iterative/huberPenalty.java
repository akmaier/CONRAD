package edu.stanford.rsl.conrad.reconstruction.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;

public class huberPenalty {

	public float beta = 0.1f;
	public float delta = 0.01f; 

	public huberPenalty( float beta, float delta ){
		this.beta = beta;
		this.delta = delta;
	}

	public float huber( Grid3D u ){

		float R = 0.0f;
		float deltaSh = delta * delta / 2;

		final int maxI = u.getSize()[0];
		final int maxJ = u.getSize()[1];
		final int maxK = u.getSize()[2];

		for (int i = 0; i < maxI-1; i++) {
			for (int j = 0; j < maxJ-1; j++) {
				for (int k = 0; k < maxK; k++) {

					float d1 = Math.abs( u.getAtIndex(i, j, k) - u.getAtIndex(i+1, j, k) );
					float d2 = Math.abs( u.getAtIndex(i, j, k) - u.getAtIndex(i, j+1, k) );

					if ( d1 < delta ){
						R = R + d1 * d1 / 2; 
					}else{
						R = R + delta * d1 - deltaSh;
					}

					if ( d2 < delta ){
						R = R + d2 * d2 / 2; 
					}else{
						R = R + delta * d2 - deltaSh;
					}

				}
			}
		}

		R = R * beta;

		return R;
	}

	public void huberDerivative( Grid3D u,  Grid3D s ){

		final int maxI = u.getSize()[0];
		final int maxJ = u.getSize()[1];
		final int maxK = u.getSize()[2];

		NumericPointwiseOperators.fill(s , 0.0f );

		if (maxI != s.getSize()[0] || maxJ != s.getSize()[1]
				|| maxK != s.getSize()[2])
			System.out.print("Wrong Size! \n");

		for (int i = 0; i < maxI-1; i++) {
			for (int j = 0; j < maxJ-1; j++) {
				for (int k = 0; k < maxK; k++) {

					float d1 =  u.getAtIndex(i, j, k) - u.getAtIndex(i+1, j, k) ;
					float d2 =  u.getAtIndex(i, j, k) - u.getAtIndex(i, j+1, k) ;

					if ( d1 < delta ){
						d1 = - delta; 
					} 
					
					if ( d1 > delta ){
						d1 =  delta;
					}

					if ( d2 < delta ){
						d2 = - delta; 
					} 
					
					if ( d1 > delta ){
						d2 =  delta;
					}

					s.setAtIndex(i, j, k, ( s.getAtIndex(i, j, k) + d1 + d2 ) );
					s.setAtIndex(i+1, j, k, ( s.getAtIndex(i+1, j, k) -  d2 ) );
					s.setAtIndex(i, j+1, k, ( s.getAtIndex(i, j+1, k) -  d2 ) );

				}
			}
		}

		NumericPointwiseOperators.multiplyBy(s, beta );
		return;
	}

	public void huberCurvature( Grid3D u, Grid3D s ){

		final int maxI = u.getSize()[0];
		final int maxJ = u.getSize()[1];
		final int maxK = u.getSize()[2];

		NumericPointwiseOperators.fill(s , 0.0f );

		if (maxI != s.getSize()[0] || maxJ != s.getSize()[1]
				|| maxK != s.getSize()[2])
			System.out.print("Wrong Size! \n");

		for (int i = 0; i < maxI-1; i++) {
			for (int j = 0; j < maxJ-1; j++) {
				for (int k = 0; k < maxK; k++) {

					float d1 = Math.abs( u.getAtIndex(i, j, k) - u.getAtIndex(i+1, j, k) );
					float d2 = Math.abs( u.getAtIndex(i, j, k) - u.getAtIndex(i, j+1, k) );

					if ( d1 >  delta ){
						s.setAtIndex(i, j, k, ( s.getAtIndex(i, j, k) + 1 ) );
						s.setAtIndex(i+1, j, k, ( s.getAtIndex(i+1, j, k) -  1 ) );
					}

					if ( d2 >  delta ){
						s.setAtIndex(i, j, k, ( s.getAtIndex(i, j, k) + 1 ) );
						s.setAtIndex(i, j+1, k, ( s.getAtIndex(i, j+1, k) -  1 ) );
					}

				}
			}
		}

		NumericPointwiseOperators.multiplyBy(s, beta );
		return;
	}
	
	

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
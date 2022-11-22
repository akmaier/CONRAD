/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.tools;

public class TubenessProcessor extends HessianEvalueProcessor {

	public TubenessProcessor(boolean useCalibration) {
		this.useCalibration = useCalibration;
	}

	public TubenessProcessor(double sigma, boolean useCalibration) {
		this.sigma = sigma;
		this.useCalibration = useCalibration;
	}

	public float measureFromEvalues2D( float [] evalues ) {

		/* If either of the two principle eigenvalues is
		   positive then the curvature is in the wrong
		   direction - towards higher instensities rather than
		   lower. */

		if (evalues[1] >= 0)
			return 0;
		else
			return (float)Math.abs(evalues[1]);
	}

	public float measureFromEvalues3D( float [] evalues ) {

		/* If either of the two principle eigenvalues is
		   positive then the curvature is in the wrong
		   direction - towards higher instensities rather than
		   lower. */

		if ((evalues[1] >= 0) || (evalues[2] >= 0))
			return 0;
		else
			return (float)Math.sqrt(evalues[2] * evalues[1]);
	}
}
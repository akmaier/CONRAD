package edu.stanford.rsl.conrad.dimreduction.utils;


import java.io.IOException;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.LagrangianOptimization;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
public class PlotIterationsError {

	private String filename = "";

	public PlotIterationsError() {
	}
	public PlotIterationsError(String filename){
		this.filename = filename;
	}


	/**
	 * fetches the errors and the number of iterations from the gradient descent
	 * from the PointCloudViewableFunction and plots them into a "number of
	 * iterations-error" plot
	 * 
	 * @throws IOException
	 */
	public void computePlot_time_error(DimensionalityReduction optimization,
			LagrangianOptimization laop) throws IOException {
		double[] iterations = laop.getIterations();
		double[] error = laop.getErrors();
		PlotHelper.plot2D(iterations, error);

		if (filename.length() != 0) {
			FileHandler.save(iterations, error, filename);
		}

	}

}
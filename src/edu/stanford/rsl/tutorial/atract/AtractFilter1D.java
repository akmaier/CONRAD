package edu.stanford.rsl.tutorial.atract;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.fitting.PolynomialFunction;

/**
 * This class combines the LaplacianKernel1D, the BorderRemoval1D and the residual AtractKernel1D
 * in order to enable reconstruction on truncated data
 * @author Marco Boegel (Reco Project 2012 - Individual Project)
 *
 */
public class AtractFilter1D {

	/**
	 * This method applies laplacian filtering, collimator border removal and atract filtering.
	 * @param input Sinogram
	 */
	public void applyToGrid(Grid2D input) {

		int maxSIndex = input.getSize()[0];

		LaplaceKernel1D lap = new LaplaceKernel1D(maxSIndex);
		lap.applyToGrid(input);
		input.show("Laplace");

		BorderRemoval1D bord = new BorderRemoval1D();
		bord.applyToGridLap1D(input);
		input.show("BorderRemoval");

		AtractKernel1D atract = new AtractKernel1D(maxSIndex);
		atract.applyToGrid(input);
		input.show("Atract");

	}

	/**
	 * This method applies laplacian filtering, collimator border removal using a binary mask and atract filtering.
	 * @param input Sinogram
	 * @param collimatorMask or equivalent mask for border removal
	 */
	public void applyToGrid(Grid2D input, Grid2D collimatorMask) {

		int maxSIndex = input.getSize()[0];

		LaplaceKernel1D lap = new LaplaceKernel1D(maxSIndex);
		lap.applyToGrid(input);

		//collimatorMask.show("Mask Image");
		Grid2D tmp = new Grid2D(input);
		tmp.show("Laplace");

		/*		// Set to neighbouring pixel if exists
		for (int i = 0; i < collimatorMask.getSize()[0]; ++i)
		{
			for (int j = 0; j < collimatorMask.getSize()[1]; ++j)
			{
				if (collimatorMask.getAtIndex(i, j)==0)
				{
					if (j > 0 && j < collimatorMask.getSize()[1] - 1)
					{
						if (input.getAtIndex(i, j) < 0)
						{
							input.setAtIndex(i,j,input.getAtIndex(i, j-1));
						}
						else {
							input.setAtIndex(i,j,input.getAtIndex(i, j+1));
						}
					}
					else
						input.setAtIndex(i, j, 0.f);
				}
			}
		}*/

		NumericPointwiseOperators.multiplyBy(input, collimatorMask);

		Grid2D tmp2 = new Grid2D(input);
		tmp2.show("Laplace Removed");

		AtractKernel1D atract = new AtractKernel1D(maxSIndex);
		atract.applyToGrid(input);

		/*
		// Martin: Extend the mask and apply to laplace without removed edges!
		for (int i = 0; i < collimatorMask.getSize()[0]; ++i) {
			boolean applyCorrection = false;
			int startIndex = 0;
			for (int j = 0; j < collimatorMask.getSize()[1]; ++j) {
				if (collimatorMask.getAtIndex(i, j)==0) {
					applyCorrection = true;
					startIndex=j;
					break;
				}
			}
			if (applyCorrection){
				for (int j = 0; j < collimatorMask.getSize()[1]; ++j) {
					if (j >= startIndex - 10){
						input.setAtIndex(i, j, 0.f);
					}
				}
			}
		}
		*/

		// Andreas' atract repair code
		Grid1D regressionTemplate = new Grid1D(input.getSubGrid(0));
		float boundaryValue = input.getAtIndex(0, 0);

		// Set to neighbouring pixel if exists
		for (int i = 0; i < collimatorMask.getSize()[0]; ++i) {
			boolean applyCorrection = false;
			int startIndex = 0;
			for (int j = 0; j < collimatorMask.getSize()[1]; ++j) {
				if (collimatorMask.getAtIndex(i, j) == 0) {
					applyCorrection = true;
					startIndex = j;
					break;
				}
			}
			if (applyCorrection) {
				PolynomialFunction func = new PolynomialFunction();
				double[] inputArray = new double[startIndex];
				double[] outputArray = new double[startIndex];
				double[] tmpArray = new double[startIndex];
				for (int j = 0; j < startIndex; ++j) {
					tmpArray[j] = input.getAtIndex(i, j);
					outputArray[j] = input.getAtIndex(i, j) - regressionTemplate.getAtIndex(j);
					inputArray[j] = j;
				}
				//if (i==138) VisualizationUtil.createPlot("Original", tmpArray).show();
				//if (i==138) VisualizationUtil.createPlot("Substraction", outputArray).show();
				func.setDegree(5);
				if (startIndex > 5) {
					func.fitToPoints(inputArray, outputArray);
				}
				//if (i==138) VisualizationUtil.createScatterPlot("Detector " + i, inputArray, outputArray).show();
				for (int j = 0; j < regressionTemplate.getBuffer().length; j++) {
					if (j < startIndex) {
						input.setAtIndex(i, j, input.getAtIndex(i, j) - (float) func.evaluate(j));

						outputArray[j] = input.getAtIndex(i, j);
					} else {
						input.setAtIndex(i, j, 5f * boundaryValue);
					}
				}
				//if (i==138) VisualizationUtil.createPlot("Corrected", outputArray).show();
			}
		}

		NumericPointwiseOperators.addBy(input, -boundaryValue / 2);

		input.show("Laplace After Mask");

	}

}
/*
 * Copyright (C) 2010-2014  Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2017 Andreas Maier, Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.dimreduction.utils;


import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.WeightedInnerProductObjectiveFunction;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;

public class PlotParameterKError {
	DimensionalityReduction dimRed;

	private String project = "";

	private double interval = 10;
	private double steplength = 0.5;

	private double para;
	private String parameter;
	private double end = 1.0;
	private double start = 1.0;
	private double step = 1.0;

	private String scatterFilenameValues;
	private String scatterFilenameMin;

	private int counterP = 0;

	private int versionWIPOF = 2;

	/**
	 * Constructor of the PlotParameterKError. Builds a Parameter-K-Error-Plot with the given parameters
	 * @param project plot available for "Cube" and "SwissRoll"
	 * @param version version of the WeightedInnerProductObjectiveFunction
	 * @param intervalEndK end of the inverval of k (always starts at 0)
	 * @param steplengthK steplength of k within the  interval
	 * @param parameter parameter to be varied: available for Cube: "edgeLength", "numberOfPoints", "sd", "dimension". for SwissRoll: "numPointsSwiss", "thickness", "gap"
	 * @param startParameter start of the interval of the parameter
	 * @param endParameter end of the interval of the parameter
	 * @param stepParameter steplength of the parameter
	 */
	public PlotParameterKError(String project, int version, double intervalEndK, double steplengthK, String parameter, double startParameter, double endParameter, double stepParameter) {
		if (project == "Cube" || project == "SwissRoll") {
			this.project = project;
		}
		this.versionWIPOF = version; 
		this.interval = intervalEndK; 
		this.steplength = steplengthK;
		this.parameter = parameter;
		this.start = startParameter;
		para = startParameter;
		this.end = endParameter;
		this.step = stepParameter;
		scatterFilenameValues = "values";
		scatterFilenameMin = "min";
	}
	/**
	 * Constructor of the PlotParameterKError. Builds a Parameter-K-Error-Plot with the given parameters
	 * @param project plot available for "Cube" and "SwissRoll"
	 * @param version version of the WeightedInnerProductObjectiveFunction
	 * @param intervalEndK end of the inverval of k (always starts at 0)
	 * @param steplengthK steplength of k within the  interval
	 * @param parameter parameter to be varied: available for Cube: "edgeLength", "numberOfPoints", "sd", "dimension". for SwissRoll: "numPointsSwiss", "thickness", "gap"
	 * @param startParameter start of the interval of the parameter
	 * @param endParameter end of the interval of the parameter
	 * @param stepParameter steplength of the parameter
	 * @param filename if the filename is set the computed results of the plot will be saved in a txt file, (give filename without .txt!)
	 */
	public PlotParameterKError(String project, int version, double intervalEndK, double steplengthK, String parameter, double startParameter, double endParameter, double stepParameter, String filename) {
		if (project == "Cube" || project == "SwissRoll") {
			this.project = project;
		} else {
			System.out.println("You can only make a Parameter-K-Error plot with the 'Cube' or the 'SwissRoll'!");
			return; 
		}
		this.versionWIPOF = version; 
		this.interval = intervalEndK; 
		this.steplength = steplengthK;
		this.parameter = parameter;
		this.start = startParameter;
		para = startParameter;
		this.end = endParameter;
		this.step = stepParameter;
		scatterFilenameValues = filename + "_values";
		scatterFilenameMin = filename + "_min";
	}



	/**
	 * function to set the DimensionalityReduction
	 * 
	 * @param dimRed
	 *            Dimensionality Reduction of the actual computation
	 */
	public void setDimensionalityReduction(DimensionalityReduction dimRed) {
		this.dimRed = dimRed;
		dimRed.setProject(this.project);
	}

	/**
	 * setter function to set the parameter to be changed
	 * @param parameter parameter to be changed
	 */
	protected void setParameter(double parameter) {
		this.para = parameter;
	}


	/**
	 * Computes  the 3Dplot "parameter-k-error"
	 * 
	 * @throws Exception
	 */
	public void computePlot_parameter_k_error() throws Exception {
		if (this.project.length() == 0) {
			throw new MyException(
					"You have to choose a Project (Cube or SwissRoll)");
		} else {

			double[] errors = new double[(int) Math.ceil((double) interval
					/ steplength)];
			double[] k = new double[(int) Math.ceil((double) interval
					/ steplength)];

			updateParameter(para);

			dimRed.computeCoordinates(dimRed.getNumPointsSwiss(),
					dimRed.getThickness(), dimRed.getGap(), dimRed.getDim(),
					dimRed.getNumPoints(), dimRed.getEdgeLength(),
					dimRed.getSd(), dimRed.getSaveCoordinates(),
					dimRed.getWithsavedCoordinates());
			double[][] distanceMatrix = dimRed.getDistances();
			int counter = 0;
			for (double i = 0; i < interval; i += steplength) {
				PointCloudViewableOptimizableFunction functionScatter = new WeightedInnerProductObjectiveFunction(versionWIPOF, i);
				functionScatter.setDistances(distanceMatrix);
				((WeightedInnerProductObjectiveFunction) functionScatter)
						.setDistances(distanceMatrix);
				functionScatter.setOptimization(dimRed);
				FunctionOptimizer func2 = new FunctionOptimizer();
				int newDim = distanceMatrix.length
						* dimRed.getTargetDimension();
				func2.setDimension(newDim);
				double[] initial2 = new double[newDim];
				for (int j = 0; j < initial2.length; j++) {
					initial2[j] = Math.random() - 0.5;

				}

				func2.setConsoleOutput(false);

				func2.setInitialX(initial2);
				func2.setOptimizationMode(OptimizationMode.Function);
				func2.optimizeFunction(functionScatter);
				double[] result2 = func2.getOptimum();
				Error er = new Error();
				double error = er.computeError(HelperClass.wrapArrayToPoints(
						result2, distanceMatrix.length), "", distanceMatrix);

				errors[counter] = error;
				k[counter] = i;
				++counter;

			}

			int length = (int) ((end - start) / step);

			ScatterPlot scatterPlotValues = dimRed.getScatterPlotValues();
			if (scatterPlotValues == null) {
				scatterPlotValues = new ScatterPlot(length
						* (int) (interval / steplength));
			}

			double[] parameterArray = new double[(int) (interval / steplength)];
			for (int i = 0; i < (int) (interval / steplength); ++i) {
				parameterArray[i] = para;
			}

			scatterPlotValues.addValue(parameterArray, k, errors,
					scatterFilenameValues);

			ScatterPlot scatterPlotMin = dimRed.getScatterPlotMin();

			if (scatterPlotMin == null) {
				scatterPlotMin = new ScatterPlot(length);
			}
			int indexMin = scatterPlotMin.findMinimum(errors);

			scatterPlotMin.addValue(para, k[indexMin], errors[indexMin],
					scatterFilenameMin);

			if (this.counterP < length - 1) {
				++this.counterP;

				para += step;
				this.setParameter(para);
				this.updateParameter(para);
				dimRed.computeCoordinates(dimRed.getNumPointsSwiss(),
						dimRed.getThickness(), dimRed.getGap(),
						dimRed.getDim(), dimRed.getNumPoints(),
						dimRed.getEdgeLength(), dimRed.getSd(),
						dimRed.getSaveCoordinates(),
						dimRed.getWithsavedCoordinates());
				dimRed.setScatterPlotValues(scatterPlotValues);
				dimRed.setScatterPlotMin(scatterPlotMin);
				computePlot_parameter_k_error();

			}
		}

	}
	/**
	 * updates the parameters during the computation of the 3Dplot
	 * @param actualPara value of the new parameter, the parameter is saved in the String parameter
	 * @throws MyException
	 */
	public void updateParameter(double actualPara) throws MyException {
		if (parameter == "dimension") {
			dimRed.setdim((int) actualPara);
		} else if (parameter == "numberOfPoints") {
			dimRed.setNumPoints((int) actualPara);
		} else if (parameter == "edgeLength") {
			dimRed.setEdgeLength(actualPara);
		} else if (parameter == "sd") {
			dimRed.setSd(actualPara);
		} else if (parameter == "numPointsSwiss") {
			dimRed.setNumPointsSwiss((int) actualPara);
		} else if (parameter == "thickness") {
			dimRed.setThickness((int) actualPara);
		} else if (parameter == "gap") {
			dimRed.setGap(actualPara);
		} else {
			throw new MyException("You have to choose a parameter!");
		}
	}

}
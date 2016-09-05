package edu.stanford.rsl.tutorial.dmip;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.Map.Entry;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;
import edu.stanford.rsl.jpop.OptimizableFunction;
import edu.stanford.rsl.jpop.OptimizationOutputFunction;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;
import ij.IJ;
import ij.ImageJ;
import ij.gui.PlotWindow;

/**
 * Exercise 7 of Diagnostic Medical Image Processing (DMIP)
 * Solve the Registration Problem using Sum of Squared Difference
 * @author Bastian Bier
 *
 */
public class Registration3 {

	public Grid2D reference = null; 
	public Grid2D image = null;
	
	public Grid3D movingStack = null;
	
	public int iteration = 0;
	double[] saveParameters;

	public class CostFunction implements OptimizableFunction, OptimizationOutputFunction{


		private TreeMap<Integer,Double> resultVisualizer;

		private PlotWindow resultVisualizerPlot;
		
		@Override
		public void setNumberOfProcessingBlocks(int number) {
		}

		@Override
		public int getNumberOfProcessingBlocks() {
			return 1;
		}

		/*
		 * This function gets a parameter vector and returns the result of the cost function
		 */
		@Override
		public double evaluate(double[] x, int block) {
			int nrChanges = 0;
			for (int i = 0; i < x.length; i++) {
				if(saveParameters[i]!=x[i]){
					nrChanges++;
				}
			}

			// Define Rotation
			SimpleMatrix r = new SimpleMatrix(2,2);
			double phi2 = x[0] * (2*Math.PI)/360;
			r.setElementValue(0, 0, Math.cos(phi2));
			r.setElementValue(0, 1, - Math.sin(phi2));
			r.setElementValue(1, 0, Math.sin(phi2));
			r.setElementValue(1, 1, Math.cos(phi2));

			// Define translation
			double t_x = x[1];
			double t_y = x[2];

			Grid2D im_tmp = new Grid2D(image);
			
			// Perform rotation/translation
			SimpleVector t = new SimpleVector(t_x,t_y);
			AffineTransform affine = new AffineTransform(r, t);
			im_tmp.applyTransform(affine);
			
			if(nrChanges>=3){
				movingStack.setSubGrid(iteration, im_tmp);
				iteration++;
			}
			
			// Calculate the cost function
			double cost = SumOfSquaredDifferences(reference, im_tmp);
			System.arraycopy(x, 0, saveParameters, 0, x.length);
			
			return cost;
		}

		@Override
		public void optimizerCallbackFunction(int currIterationNumber, double[] x, double currFctVal,
				double[] gradientAtX) {
			// Visualization of cost function value over time
			if (this.resultVisualizer == null)
				resultVisualizer = new TreeMap<Integer, Double>();
			resultVisualizer.put(currIterationNumber, currFctVal);
			if (resultVisualizerPlot != null)
				resultVisualizerPlot.close();

			Grid1D out = new Grid1D(resultVisualizer.size());
			Iterator<Entry<Integer,Double>> it = resultVisualizer.entrySet().iterator();
			while (it.hasNext()) {
				Entry<Integer,Double> e = it.next();
				out.setAtIndex(e.getKey(), e.getValue().floatValue());
			}
			resultVisualizerPlot = VisualizationUtil.createPlot(out.getBuffer()).show();
		}

	}
	
	private double SumOfSquaredDifferences(Grid2D ref, Grid2D imageMoving){

		double sum = 0.0; 

		for(int i = 0; i < ref.getWidth(); i++)
		{
			for(int j = 0; j < ref.getHeight(); j++)
			{
				// TODO: calculate SSD
			}
		}

		return sum/ref.getNumberOfElements();
	}
	
	public double[] performOptimization(){
		
		// Initialize optimization class
		FunctionOptimizer fo = new FunctionOptimizer();
		fo.setDimension(3);
		fo.setItnlim(50);
		movingStack = new Grid3D(reference.getWidth(), reference.getHeight(), 1000, false);
		iteration = 0;
		fo.setOptimizationMode(OptimizationMode.Function);
		fo.setNdigit(8);
		fo.setMsg(16);
		fo.setInitialX(new double[]{0,0,0});
		fo.setMaxima(new double[]{50,50,50});
		fo.setMinima(new double[]{-50,-50,-50});

		// Initialize the Costfunction of the optimization
		CostFunction cf = new CostFunction();
		saveParameters = new double[]{Double.POSITIVE_INFINITY,Double.POSITIVE_INFINITY,Double.POSITIVE_INFINITY};
		
		// Optimization visualized
		ArrayList<OptimizationOutputFunction> visFcts = new ArrayList<OptimizationOutputFunction>();
		visFcts.add(cf);
		fo.setCallbackFunctions(visFcts);
		
		// Perform the optimization with the given cost function
		double[] result = fo.optimizeFunction(cf);
		
		return result;
	}


	

	public static void main(String[] args){

		ImageJ ij = new ImageJ();

		///////////////////////////////////////////////////////
		// Part 1: Apply a transformation on a phantom image //
		///////////////////////////////////////////////////////
		
		// Create Phantom
		Grid2D phantom = new SheppLogan(256);
		
		// Set the Origin of the image in its center
		// The default origin of an image is in its top left corner
		// Default Origin: [0.0, 0.0]
		int w = phantom.getWidth();
		int h = phantom.getHeight();
		phantom.setOrigin(-(w-1) / 2 , -(h-1)/2);
		
		Grid2D phantom_blurred = new Grid2D(phantom);
		IJ.run(ImageUtil.wrapGrid(phantom_blurred,""),"Gaussian Blur...", "sigma=3");
		phantom.show("Phantom");
		
		// Rotate the phantom by 45° and translate it with t = [20, 1]

		// Define Rotation and translation
		SimpleMatrix r = new SimpleMatrix(2,2);
		
		// TODO: set phi
		double phi = 0;
		
		// TODO: fill the rotation matrix
		// TODO
		// TODO
		// TODO
		// TODO

		// TODO: define translation
		SimpleVector t = new SimpleVector(0,0);

		// Initialize transformed phantom
		Grid2D transformedPhantom = new Grid2D(phantom);
		Grid2D transformedPhantom_blurred = new Grid2D(phantom_blurred);

		// Create the affine transformation
		AffineTransform affine = new AffineTransform(r, t);
		
		// Apply the transformation
		transformedPhantom.applyTransform(affine);
		transformedPhantom_blurred.applyTransform(affine);

		transformedPhantom.show("Transformed Phantom");
		
		
		/////////////////////////////////////
		// Part 2: Find the transformation //
		/////////////////////////////////////
		
		// Registration of the transformed image to the initial phantom
		Registration3 reg3 = new Registration3();
		reg3.reference = phantom_blurred;
		reg3.image = transformedPhantom_blurred;
		
		// Optimization
		double[] res = reg3.performOptimization();
		
		// Stack for visualization purposes only
		Grid3D optimizationStepsGrid = new Grid3D(reg3.reference.getWidth(),reg3.reference.getHeight(),reg3.iteration,false);
		for (int i = 0; i < optimizationStepsGrid.getSize()[2]; i++) {
			optimizationStepsGrid.setSubGrid(i, reg3.movingStack.getSubGrid(i));
		}
		optimizationStepsGrid.show();
		
		// Transform image back
		Grid2D backtransformedImage = new Grid2D(transformedPhantom);
		phi = (2*Math.PI)/360 * res[0];
		r.setElementValue(0, 0, Math.cos(phi));
		r.setElementValue(0, 1, - Math.sin(phi));
		r.setElementValue(1, 0, Math.sin(phi));
		r.setElementValue(1, 1, Math.cos(phi));

		SimpleVector t2 = new SimpleVector(res[1],res[2]);
		AffineTransform affine2 = new AffineTransform(r, t2);
		backtransformedImage.applyTransform(affine2);
		backtransformedImage.show();
	}
}

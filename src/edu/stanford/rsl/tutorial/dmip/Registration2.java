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
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.OptimizableFunction;
import edu.stanford.rsl.jpop.OptimizationOutputFunction;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;

import ij.IJ;
import ij.ImageJ;
import ij.gui.PlotWindow;


/**
 * Exercise 7 of Diagnostic Medical Image Processing (DMIP)
 * Using Mutual Information to solve the registration problem
 * @author Bastian Bier
 *
 */
public class Registration2 {
	
	public Grid2D reference = null; 
	public Grid2D image = null;
	
	public Grid3D movingStack = null;
	
	public int iteration = 0;
	double[] saveParameters;
	
	class CostFunction implements OptimizableFunction, OptimizationOutputFunction{

		private TreeMap<Integer,Double> resultVisualizer;

		private PlotWindow resultVisualizerPlot;
		
		@Override
		public void setNumberOfProcessingBlocks(int number) {
		}

		@Override
		public int getNumberOfProcessingBlocks() {
			return 1;
		}

		@Override
		public double evaluate(double[] x, int block) {
			
			int nrChanges = 0;
			for (int i = 0; i < x.length; i++) {
				if(saveParameters[i]!=x[i]){
					nrChanges++;
				}
			}
			
			// Define Rotation
			SimpleMatrix r = new SimpleMatrix(2, 2);
			double phi2 = x[0] * (2 * Math.PI) / 360;
			r.setElementValue(0, 0, Math.cos(phi2));
			r.setElementValue(0, 1, -Math.sin(phi2));
			r.setElementValue(1, 0, Math.sin(phi2));
			r.setElementValue(1, 1, Math.cos(phi2));

			// Define translation
			double t_x = x[1];
			double t_y = x[2];

			Grid2D im_tmp = new Grid2D(image);

			// Perform rotation/translation
			SimpleVector t = new SimpleVector(t_x, t_y);
			AffineTransform affine = new AffineTransform(r, t);
			im_tmp.applyTransform(affine);

			if (nrChanges >= 3) {
				movingStack.setSubGrid(iteration, im_tmp);
				iteration++;
			}

			// Calculate the cost function
			double cost = calculateMutualInformation(reference, im_tmp);
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
	
	/**
	 * Method to calculate the Mutual Information
	 * @param ref reference image
	 * @param mov moving image
	 * @return negative mutual information
	 */
	private double calculateMutualInformation(Grid2D ref, Grid2D mov){
		
		int histSize = 256;
		
		// Step 1: Calculate joint histogram
		SimpleMatrix jointHistogram = calculateJointHistogram(ref, mov);
		
		// Step 2: Get histogram for a single image from the joint histogram
		// a) for the first image
		SimpleVector histo1 = new SimpleVector(histSize);
		histo1 = getHistogramFromJointHistogram(jointHistogram);
		
		// b) for the second image
		SimpleVector histo2 = new SimpleVector(histSize);
		SimpleMatrix jh_t = jointHistogram.transposed();
		histo2 = getHistogramFromJointHistogram(jh_t);
		
		// Step 3: Calculate the marginal entropies and the joint entropy
		double entropy_jointHisto = 0;
		double entropy_histo1 = 0;
		double entropy_histo2 = 0;
		
		for(int i = 0; i < histSize; i++)
		{
			if(histo1.getElement(i) != 0)
			{
				// TODO: calculate entropy for histogram 1
			}
			
			if(histo2.getElement(i) != 0)
			{
				// TODO: calculate entropy for histogram 2
			}
		}
		
		for (int i = 0; i < histSize; i++) 
		{
			for (int j = 0; j < histSize; j++)
			{
				if(jointHistogram.getElement(i, j) != 0)
				{
					// TODO: calculate entropy of the joint histogram
				}
			}
		}
	
		// make sure to consider the - in from of the sum (Entropy formula)
		// TODO
		// TODO
		// TODO
		
		// Step 4: Calculate the mutual information
		// Note: The mutual information is high for a good match
		// but we require a minimization problem --> the result is inverted to fit the optimizer
		double mutual_information = 0;
		// TODO: calculate the mutual information
		
		return mutual_information * 1000;
	}
	
	/**
	 * Method to calculate the joint histogram of two images
	 * @param im1 image1
	 * @param im2 image2
	 * @return a SimpleMatrix corresponding to the joint histogram
	 */
	private SimpleMatrix calculateJointHistogram(Grid2D im1, Grid2D im2){
		
		// Calculate joint histogram
		int histSize = 256;
		SimpleMatrix jH = new SimpleMatrix(histSize, histSize);

		for (int i = 0; i < histSize; i++) {
			for (int j = 0; j < histSize; j++) {
				// TODO
			}
		}
		
		// Divide by the number of elements in order to get probabilities
		for (int i = 0; i < histSize; i++) {
			for (int j = 0; j < histSize; j++) {
				// TODO
			}
		}
		
		return jH;
	}
	
	/**
	 * Method to calculate a histogram from a joint histogram
	 * @param jH The joint histogram
	 * @return a SimpleVector corresponding to the marginal histogram
	 */
	private SimpleVector getHistogramFromJointHistogram(SimpleMatrix jH){

		// Calculate histogram from joint histogram
		int histSize = 256;
		SimpleVector hist = new SimpleVector(histSize);
		hist.zeros();
		
		for(int i = 0; i < histSize; i++)
		{
			for(int j = 0; j < histSize; j++)
			{
				// TODO: sum up over the columns
			}
		}
		
		return hist;
	}
	
	private double[] performOptimization(){
		
		FunctionOptimizer fo = new FunctionOptimizer();
		
		fo.setDimension(3);
		fo.setNdigit(6);
		fo.setItnlim(50);
		fo.setMsg(16);
		fo.setInitialX(new double[]{0,0,0});
		fo.setMaxima(new double[]{50,50,50});
		fo.setMinima(new double[]{-50,-50,-50});
		fo.setOptimizationMode(OptimizationMode.Function);
		
		CostFunction cF = new CostFunction();
		
		movingStack = new Grid3D(reference.getWidth(), reference.getHeight(), 1000, false);
		iteration = 0;
		saveParameters = new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
		
		// Optimization visualized
		ArrayList<OptimizationOutputFunction> visFcts = new ArrayList<OptimizationOutputFunction>();
		visFcts.add(cF);
		fo.setCallbackFunctions(visFcts);
		
		double result[] = fo.optimizeFunction(cF);
		
		return result;
	}
	
	public static void main(String[] args){
		
		ImageJ ij = new ImageJ();
		
		// Load images
		// TODO Adjust paths
		String filename1 = "C:/StanfordRepo/CONRAD/src/edu/stanford/rsl/tutorial/dmip/T1.png";
		String filename2 = "C:/StanfordRepo/CONRAD/src/edu/stanford/rsl/tutorial/dmip/Proton.png";
		
		Grid2D image1 = ImageUtil.wrapImagePlus(IJ.openImage(filename1)).getSubGrid(0);
		Grid2D image2 = ImageUtil.wrapImagePlus(IJ.openImage(filename2)).getSubGrid(0);
		
		image1.show("Input Image 1");
		image2.show("Input Image 2");
		
		// Set the Origin of the image in its center
		// The default origin of an image is in its top left corner
		// Default Origin: [0.0, 0.0]
		int w = image1.getWidth();
		int h = image1.getHeight();

		image1.setOrigin(-(w-1) / 2 , -(h-1)/2);
		image2.setOrigin(-(w-1) / 2 , -(h-1)/2);
		image1.setSpacing(1);
		image2.setSpacing(1);
		
		// Blurred Images for the registration to avoid local minima during optimization
		Grid2D image1_blurred = new Grid2D(image1);
		Grid2D image2_blurred = new Grid2D(image2);
		
		IJ.run(ImageUtil.wrapGrid(image1_blurred,""),"Gaussian Blur...", "sigma=4");
		IJ.run(ImageUtil.wrapGrid(image2_blurred,""),"Gaussian Blur...", "sigma=4");
		
		Registration2 reg2 = new Registration2();
		reg2.reference = image1_blurred;
		reg2.image = image2_blurred;
		
		// Perform Optimization
		double res[] = reg2.performOptimization();
		
		// Stack for visualization purposes only
		Grid3D optimizationStepsGrid = new Grid3D(reg2.reference.getWidth(), reg2.reference.getHeight(), reg2.iteration, false);
		for (int i = 0; i < optimizationStepsGrid.getSize()[2]; i++) {
			optimizationStepsGrid.setSubGrid(i, reg2.movingStack.getSubGrid(i));
		}
		optimizationStepsGrid.show("Optimization Steps");
		
		// Transform image back
		SimpleMatrix r = new SimpleMatrix(2,2);
		Grid2D registeredImage = new Grid2D(image2);
		double phi = (2 * Math.PI) / 360 * res[0];
		r.setElementValue(0, 0, Math.cos(phi));
		r.setElementValue(0, 1, -Math.sin(phi));
		r.setElementValue(1, 0, Math.sin(phi));
		r.setElementValue(1, 1, Math.cos(phi));

		SimpleVector t2 = new SimpleVector(res[1], res[2]);
		AffineTransform affine2 = new AffineTransform(r, t2);
		registeredImage.applyTransform(affine2);
		registeredImage.show("Registered Image");		
	}
}
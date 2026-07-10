package edu.stanford.rsl.tutorial.mipda;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.Map.Entry;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
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


/**
 * Using mutual information to solve the registration problem
 * Programming exercise for chapter "Rigid Registration"
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Frank Schebesch, Bastian Bier, Anna Gebhard, Mena Abdelmalek, Ashwini Jadhav
 *
 */

public class ExerciseMI {
	
	final double blurSigma = 4.0;
	final int iterLimit = 50; 
	
	final Grid2D image1;
	final Grid2D image2;
	
	public Grid2D reference_img; 
	public Grid2D moving_img;
	
	public Grid3D movingStack;

	
	public static void main(String[] args){
		
		ImageJ ij = new ImageJ();
		
		ExerciseMI exObj = new ExerciseMI();
		
		exObj.image1.show("Input Image 1");
		exObj.image2.show("Input Image 2");
		
		// both images are blurred to avoid the optimizer running into local minima
		// registering the blurred versions supposedly result in the same transformation parameters
		exObj.reference_img.show("Reference image");
		exObj.moving_img.show("Moving image");
		
		// initialize cost function
		CostFunction costFunction = exObj.new CostFunction();
		
		/* Hint for TODO-s:
		 * The cost function computes the difference of a certain measure for both images. 
		 * You can look up the details in the respective nested class below if you want to,
		 * otherwise we spare you the details and point you right to the tasks.
		 * 
		 * TASK: You have to implement the methods that are required to compute the similarity measure,
		 * which is mutual information here (more details are in the course description).
		 * These methods are
		 * - double calculateMutualInformation(Grid2D, Grid2D){} [lines 106-166]
		 * - SimpleMatrix calculateJointHistogram(Grid2D, Grid2D, int){} [lines 175-211]
		 * - SimpleVector getHistogramFromJointHistogram(Grid2D,boolean){} [lines 219-246]
		 */

		// perform registration
		double res[] = exObj.performRegistration(costFunction);
		
		// visualization of cost function value over time
		VisualizationUtil.createPlot(costFunction.getCostPerIterGrid().getBuffer()).show();
		
		// stack for visualization purposes only
		if (exObj.movingStack.getNumberOfElements() > 0)
			exObj.movingStack.show("Optimization Steps");
		
		// transform image back
		double phi = Math.PI/180*res[0];
		SimpleVector t = new SimpleVector(res[1], res[2]);
		RigidTransform rigidTransform = exObj.new RigidTransform(phi,t);
		
		// show that the found transform registers the high-resolution moving image image2 well
		Grid2D registeredImage = new Grid2D(exObj.image2);
		registeredImage.applyTransform(rigidTransform);
		registeredImage.show("Registered Image");	
	}

	/**
	 * Method to calculate the mutual information (MI)
	 * @param ref reference image
	 * @param mov moving image
	 * @return negative mutual information
	 */
	// TODO-s here!
	public double calculateMutualInformation(Grid2D ref, Grid2D mov){
		
		final int histSize = 256;
		
		// Step 1: Calculate joint histogram
		SimpleMatrix jointHistogram = calculateJointHistogram(ref, mov, histSize);
		
		// Step 2: Get histogram for a single image from the joint histogram
		// a) for the first image
		SimpleVector histo1 = new SimpleVector(histSize);
		histo1 = getHistogramFromJointHistogram(jointHistogram,false);
		
		// b) for the second image
		SimpleVector histo2 = new SimpleVector(histSize);
		histo2 = getHistogramFromJointHistogram(jointHistogram,true);
		
		// Step 3:
		// a) Calculate the joint entropy
		double entropy_jointHisto = 0;
		
		for (int i = 0; i < histSize; i++) {
			
			for (int j = 0; j < histSize; j++){
				
				if(jointHistogram.getElement(i, j) != 0) {
					
					// calculate entropy of the joint histogram (hint: use logarithm base 2 and use the correct sign)
					entropy_jointHisto = 1;//TODO
				}
			}
		}
				
		// b) Calculate the marginal entropies
		double entropy_histo1 = 0;
		double entropy_histo2 = 0;
		
		for(int i = 0; i < histSize; i++){
			
			if(histo1.getElement(i) != 0){
				
				// calculate entropy for histogram 1 (hint: use logarithm base 2 and use the correct sign)
				entropy_histo1 = 1;//TODO
			}
			
			if(histo2.getElement(i) != 0){
				
				// calculate entropy for histogram 2 (hint: use logarithm base 2 and use the correct sign)
				entropy_histo2 = 1;//TODO
			}
		}

		// Note: Make sure that you considered the correct sign in the entropy formulas!
		
		// Step 4: Calculate the mutual information
		double mutual_information = entropy_histo1 + entropy_histo2 - entropy_jointHisto;
		
		// Mutual information is high for a good match.
		// For the optimizer we require a minimization problem.
		// The factor 100 serves as a trick to stabilize the optimization process.
		return mutual_information *= -100;
	}
	
	/**
	 * Method to calculate the joint histogram of two images
	 * @param im1 image1
	 * @param im2 image2
	 * @return a SimpleMatrix corresponding to the joint histogram
	 */
	// TODO-s here!
	public SimpleMatrix calculateJointHistogram(Grid2D im1, Grid2D im2, int histSize){
		
		SimpleMatrix jH = new SimpleMatrix(histSize, histSize);
		
		int imWidth = im1.getWidth();
		int imHeight = im1.getHeight();
		
		if (imWidth != im2.getWidth() && imHeight != im2.getHeight()) {
			
			System.err.println("Image inputs have to have same size for joint histogram evaluation.");
			return jH;
		}
		
		float min1 = NumericPointwiseOperators.min(im1);
		float scaleFactor1 = (histSize - 1)/(NumericPointwiseOperators.max(im1) - min1);
		float min2 = NumericPointwiseOperators.min(im2);
		float scaleFactor2 = (histSize - 1)/(NumericPointwiseOperators.max(im2) - min2);
		
		// calculate joint histogram
		for (int i = 0; i < imWidth; i++) {
			
			for (int j = 0; j < imHeight; j++) {
				
				int value_ref = (int) (scaleFactor1*(im1.getAtIndex(i, j) - min1));
				int value_mov = (int) (scaleFactor2*(im2.getAtIndex(i, j) - min2));
				
				// jH(k,l) counts how often the intensity pair k in the reference image, and l in the moving image occurs (at the corresponding location)
				// use the correct indices and set the value for jH properly
				//TODO
			}
		}
		
		// divide by the number of elements in order to get probabilities
		//TODO
		
		return jH;
	}
	
	/**
	 * Method to calculate a histogram from a joint histogram
	 * @param jH the joint histogram
	 * @return a SimpleVector corresponding to the marginal histogram
	 */
	// TODO-s here!
	public SimpleVector getHistogramFromJointHistogram(SimpleMatrix jH, boolean sumRows){

		int numCols = jH.getCols();
		int numRows = jH.getRows();
		
		int histSize = sumRows ? numCols : numRows;
			
		SimpleVector hist = new SimpleVector(histSize);
		
		for(int i = 0; i < numRows; i++) {
			
			for(int j = 0; j < numCols; j++) {
				
				if (sumRows) {
					
					// sum up over the rows
					//TODO
				}
				else {
					
					// sum up over the columns
					//TODO
				}				
			}
		}
		
		return hist;
	}
	
	/**
	 * 
	 * end of the exercise
	 */		
	
	public ExerciseMI() {
		
		// Load images
		String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/mipda/";
		
		String filename1 = imageDataLoc + "T1.png";
		String filename2 = imageDataLoc + "Proton.png";
		
		image1 = readImageFile(filename1);
		image2 = readImageFile(filename2);

		// set the origin of the image in its center (the default origin of an image is in its top left corner)
		centerOrigin(image1,new double[]{1.0,1.0});
		centerOrigin(image2,new double[]{1.0,1.0});

		// blurred images for the registration to avoid local minima during optimization
		reference_img = blurImage(image1,blurSigma);
		moving_img = blurImage(image2,blurSigma);
		
		movingStack = new Grid3D(reference_img.getWidth(), reference_img.getHeight(), iterLimit, false);
	}
	
	public Grid2D readImageFile(String filename){
		return ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
	}
	
	void centerOrigin(Grid2D image, double[] spacing) {
		
		image.setOrigin(-(image.getWidth()-1)/2 , -(image.getHeight()-1)/2);
		image.setSpacing(spacing);
	}
	
	Grid2D blurImage(Grid2D image, double sigma) {
		
		Grid2D image_blurred = new Grid2D(image);
		IJ.run(ImageUtil.wrapGrid(image_blurred,""),"Gaussian Blur...", "sigma="+sigma);
		
		return image_blurred;
	}
	
	public double[] performRegistration(CostFunction cF){
		
		FunctionOptimizer fo = new FunctionOptimizer();

		fo.setDimension(3);
		fo.setNdigit(6);
		fo.setItnlim(iterLimit);
		fo.setMsg(16);
		fo.setInitialX(new double[]{0,0,0});
		fo.setMaxima(new double[]{50,50,50});
		fo.setMinima(new double[]{-50,-50,-50});
		fo.setOptimizationMode(OptimizationMode.Function);

		// enable reading intermediate results
		ArrayList<OptimizationOutputFunction> visFcts = new ArrayList<OptimizationOutputFunction>();
		visFcts.add(cF);
		fo.setCallbackFunctions(visFcts);
		
		double result[] = fo.optimizeFunction(cF);
		
		movingStack = cropMovingStack(movingStack, cF.getIteration());

		return result;
	}

	private Grid3D cropMovingStack(Grid3D stack, int bound) {
		
		int[] mStackSize = stack.getSize();
		
		ArrayList<Grid2D> buffer = stack.getBuffer();
		
		Grid3D newStack = new Grid3D(mStackSize[0], mStackSize[1], bound);
		for (int i=0; i<bound; i++) {
			newStack.setSubGrid(i, buffer.get(i));	
		}
		
		return newStack;
	}
	
	public class CostFunction implements OptimizableFunction, OptimizationOutputFunction{

		private TreeMap<Integer,Double> costPerIter_map;
		
		Grid2D tmp_img;
		int iteration = 0;
		int subIteration = 0; // only for debugging
		
		@Override
		public void setNumberOfProcessingBlocks(int number) {
		}

		@Override
		public int getNumberOfProcessingBlocks() {
			return 1;
		}

		@Override
		public double evaluate(double[] x, int block) {

			// define rotation and translation
			double phi = x[0]*Math.PI/180;
			SimpleVector t = new SimpleVector(x[1], x[2]);
			RigidTransform rigid = new RigidTransform(phi, t);
			
			// perform rotation/translation
			tmp_img = new Grid2D(moving_img);
			tmp_img.applyTransform(rigid);
			
			// calculate the cost function
			double cost = calculateMutualInformation(reference_img, tmp_img);
			
			subIteration++;
			return cost;
		}
		
		@Override
		public void optimizerCallbackFunction(int currIterationNumber, double[] x,
				double currFctVal, double[] gradientAtX) {
			
			iteration = currIterationNumber;
			
			// fill movingStack with current transformed image
			movingStack.setSubGrid(currIterationNumber, tmp_img);
			
			// generate cost plot data
			if (costPerIter_map == null)
				costPerIter_map = new TreeMap<Integer, Double>();
			
			costPerIter_map.put(currIterationNumber, currFctVal);
		}
		
		public Grid1D getCostPerIterGrid() {
			
			if (costPerIter_map == null)
				return new Grid1D(1);
			
			Grid1D costPerIter_grid = new Grid1D(costPerIter_map.size());
			
			Iterator<Entry<Integer,Double>> it = costPerIter_map.entrySet().iterator();
			while (it.hasNext()) {
				
				Entry<Integer,Double> e = it.next();
				costPerIter_grid.setAtIndex(e.getKey(), e.getValue().floatValue());
			}
			
			return costPerIter_grid;
		}
		
		public int getIteration() {
			return iteration;
		}
	}
	
	public class RotationMatrix2D extends SimpleMatrix {

		private static final long serialVersionUID = 6708400687838433556L;

		public RotationMatrix2D() {
			this(0.0);
		}
		
		public RotationMatrix2D(double phi) {
			
			super(2,2);
			
			this.setElementValue(0, 0, Math.cos(phi));
			this.setElementValue(0, 1, - Math.sin(phi));
			this.setElementValue(1, 0, Math.sin(phi));
			this.setElementValue(1, 1, Math.cos(phi));
		}
	} 
	
	public class RigidTransform extends AffineTransform {

		private static final long serialVersionUID = 3469069367283792094L;

		public RigidTransform(double rotationAngle, SimpleVector translationVector) {
			super(new RotationMatrix2D(rotationAngle), translationVector);
		}		
	}
	
	// getters for members
	// variables which are checked (DO NOT CHANGE!)
	public double get_blurSigma() {
		return blurSigma;
	}
	//
	public int get_iterLimit() {
		return iterLimit;
	}
	//
	public Grid2D get_image1() {
		return image1;
	}
	//
	public Grid2D get_image2() {
		return image2;
	}
	//
	public Grid2D get_reference_img() {
		return reference_img;
	}
	//
	public Grid2D get_moving_img() {
		return moving_img;
	}
	//
	public Grid3D get_movingStack() {
		return movingStack;		
	}
}
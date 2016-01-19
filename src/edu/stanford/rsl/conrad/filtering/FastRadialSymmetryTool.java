package edu.stanford.rsl.conrad.filtering;


import java.util.ArrayList;
import java.util.Arrays;

import ij.ImagePlus;
import ij.io.Opener;
import ij.plugin.filter.Convolver;
import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Tool to apply a Fast Radial Symmetry Transform on slices
 * 
 * @author Martin Berger
 *
 */
public class FastRadialSymmetryTool extends IndividualImageFilteringTool {


	/**
	 * 
	 */
	private static final long serialVersionUID = 8801374879563931231L;


	// radius of circles in pixels
	private double[] radii = new double[]{3};
	// alpha enforces the symmetry condition (higher == more symmetric)
	private double alpha = 2;
	// percentage between 0 and 1 - gradients lower than this value will be discarded
	private double smallGradientThreshold = 0;
	// flag that enforces to only use the gradient orientation and ignore the gradient magnitude
	private boolean useOrientationOnly = false;
	// flag == -1 --> compute only negative circles / flag == 1 --> compute only positive circles / otherwise --> compute both
	private int usePositiveNegativeOnly = 0;



	public FastRadialSymmetryTool (){
		configured = false;
	}


	public FastRadialSymmetryTool(double[] radii, double alpha, 
			double smallGradientThreshold, 
			ArrayList<ArrayList<double[]>> priorPoints, 
			double priorDistance){
		this.radii = radii;
		this.alpha = alpha;
		this.smallGradientThreshold = smallGradientThreshold;
		configured = true;
	}


	public void setRadii(double[] radii){
		Arrays.sort(radii, 0, radii.length-1);
		this.radii = radii;
	}

	public double[] getRadii(){
		return radii;
	}

	public void setAlpha(double alpha){
		this.alpha = alpha;
	}

	public double getAlpha(){
		return alpha;
	}

	public void setSmallGradientThreshold(double thresh){
		this.smallGradientThreshold= thresh;
	}

	public double getSmallGradientThreshold(){
		return smallGradientThreshold;
	}
	
	/**
	 * @param
	 * flag true: only use the gradient orientation / false: use orientation and magnitude
	 **/
	public void setUseOrientationOnly(boolean flag){
		this.useOrientationOnly = flag;
	}
	
	/**
	 * @return
	 * true: only use the gradient orientation / false: use orientation and magnitude
	 **/
	public boolean getUseOrientationOnly(){
		return useOrientationOnly;
	}
	
	/**
	 * @param flag 
	 * flag == -1 --> compute only negative circles / 
	 * flag == 1 --> compute only positive circles / 
	 * otherwise --> compute both
	 **/
	public void setUsePositiveNegativeOnly(int flag){
		this.usePositiveNegativeOnly = flag;
	}
	
	/**
	 * @return 
	 * -1 --> compute only negative circles / 
	 * 1 --> compute only positive circles / 
	 * otherwise --> compute both
	 **/
	public int getUsePositiveNegativeOnly(){
		return usePositiveNegativeOnly;
	}


	@Override
	public IndividualImageFilteringTool clone() {
		FastRadialSymmetryTool filter = new FastRadialSymmetryTool();
		filter.configured = configured;
		// deep copy array
		if (radii != null){
			filter.radii = new double[radii.length];
			System.arraycopy(radii, 0, filter.radii, 0, radii.length);
		}
		else{
			filter.radii = null;
		}
		filter.alpha = alpha;
		filter.smallGradientThreshold = smallGradientThreshold;
		filter.useOrientationOnly = useOrientationOnly;
		filter.usePositiveNegativeOnly = usePositiveNegativeOnly;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Fast Radial Symmetry Transform";
	}


	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		Grid2D out = new Grid2D(imageProcessor.getWidth(),imageProcessor.getHeight());

		Convolver conv = new Convolver();
		conv.setNormalize(false);
		ImageProcessor fp = ImageUtil.wrapGrid2D((Grid2D)imageProcessor.clone());
		conv.convolveFloat(fp, new float[]{-1,0,1, -2,0,2,  -1,0,1},3, 3);
		Grid2D Xgrad = ImageUtil.wrapImageProcessor(fp);

		fp = ImageUtil.wrapGrid2D((Grid2D)imageProcessor.clone());
		conv.convolveFloat(fp, new float[]{-1,-2,-1,  0,0,0,  1,2,1},3, 3);
		Grid2D Ygrad = ImageUtil.wrapImageProcessor(fp);

		Grid2D Mgrad = new Grid2D(imageProcessor.getWidth(),imageProcessor.getHeight());

		runTransform(out, Xgrad, Ygrad, Mgrad);

		return out;
	}


	private void runTransform(Grid2D output, Grid2D Xgrad, Grid2D Ygrad, Grid2D Mgrad){

		Convolver conv = new Convolver();
		conv.setNormalize(false);

		// over all radii that are in the set
		for (int i = 0; i < radii.length; i++) {
			double radius = radii[i];
			Grid2D Omap = new Grid2D(output.getWidth(),output.getHeight());
			Grid2D Mmap = useOrientationOnly ? null : new Grid2D(output.getWidth(),output.getHeight());
			Grid2D F = new Grid2D(output.getWidth(),output.getHeight());
			float kappa = (Math.round(radius) <= 1) ? 8 : 9.9f;

			// if we want to filter small gradients we need to precompute them to obtain the maximum
			float min = Float.MAX_VALUE;
			float max = Float.MIN_VALUE;
			// go over all pixels and create the gradient magnitude
			for (int y = 0; y < output.getHeight(); y++) {
				for (int x = 0; x < output.getWidth(); x++) {
					// compute gradient magnitude and normalize gradient vectors to unit length
					Mgrad.setAtIndex(x, y, (float)(Math.sqrt(Math.pow(Xgrad.getAtIndex(x, y), 2)+Math.pow(Ygrad.getAtIndex(x, y), 2))));
					if (Mgrad.getAtIndex(x, y) > max)
						max = Mgrad.getAtIndex(x, y);
					if (Mgrad.getAtIndex(x, y) < min)
						min = Mgrad.getAtIndex(x, y);
				}
			}

			// get the O and M maps
			computeOandMmap(Omap, Mmap, Xgrad, Ygrad, Mgrad, radius, kappa, max, min);

			// Unsmoothed symmetry measure at this radius value
			for (int y = 0; y < output.getHeight(); y++) {
				for (int x = 0; x < output.getWidth(); x++) {
					if (Mmap == null)
						F.setAtIndex(x, y, (float)(Math.pow(Math.abs(Omap.getAtIndex(x, y)/kappa),alpha)));
					else
						F.setAtIndex(x, y, (float)((Mmap.getAtIndex(x, y)/kappa) * Math.pow(Math.abs(Omap.getAtIndex(x, y)/kappa),alpha)));
				}
			}

			// Generate a Gaussian of size proportional to n to smooth and spread 
			// the symmetry measure.  The Gaussian is also scaled in magnitude
			// by n so that large scales do not lose their relative weighting.
			ImageProcessor ip = ImageUtil.wrapGrid2D(F);
			int gaussSize = (int)Math.round(radius);
			if(gaussSize%2==0)
				gaussSize++;
			conv.convolve(ip, ImageUtil.create2DGauss(gaussSize,gaussSize,0.25*radius).getBuffer(), gaussSize, gaussSize);
			NumericPointwiseOperators.addBy(output, ImageUtil.wrapImageProcessor(ip));

		}
	}


	private void computeOandMmap(Grid2D Omap, Grid2D Mmap, Grid2D Xgrad, Grid2D Ygrad, Grid2D Mgrad, double radius, float kappa, float max, float min){
		// go over all pixels and create the O and M maps
		for (int y = 0; y < Omap.getHeight(); y++) {
			for (int x = 0; x < Omap.getWidth(); x++) {

				if (Mgrad.getAtIndex(x, y) < CONRAD.SMALL_VALUE || 
						(smallGradientThreshold > 0 && Mgrad.getAtIndex(x, y) <= (smallGradientThreshold*(max-min)+min)) )
					continue;

				int xg = Math.round((float)radius*Xgrad.getAtIndex(x, y)/Mgrad.getAtIndex(x, y));
				int yg = Math.round((float)radius*Ygrad.getAtIndex(x, y)/Mgrad.getAtIndex(x, y));

				// compute the 'positively' and/or 'negatively' affected pixels
				if (usePositiveNegativeOnly != -1){
					int posx = correctRange(x + xg,Omap.getWidth()-1);
					int posy = correctRange(y + yg,Omap.getHeight()-1);
					Omap.setAtIndex(posx, posy, (Omap.getAtIndex(posx, posy)+1 > kappa) ? kappa : Omap.getAtIndex(posx, posy)+1);
					if (Mmap != null){
						Mmap.setAtIndex(posx, posy, Mmap.getAtIndex(posx, posy)+Mgrad.getAtIndex(x, y));
					}
				}

				if (usePositiveNegativeOnly != 1){
					int negx = correctRange(x - xg,Omap.getWidth()-1);
					int negy = correctRange(y - yg,Omap.getHeight()-1);
					Omap.setAtIndex(negx, negy, (Omap.getAtIndex(negx, negy)-1 < -kappa) ? -kappa : Omap.getAtIndex(negx, negy)-1);
					if (Mmap != null){
						Mmap.setAtIndex(negx, negy, Mmap.getAtIndex(negx, negy)-Mgrad.getAtIndex(x, y));
					}
				}
			}			
		}
	}


	// truncate value 
	private int correctRange(int val, int max){
		return ((val < 0) ? 0 : ((val > max) ? (max) : val));
	}

	@Override
	public void configure() throws Exception {
		String array = UserUtil.queryString("Enter circles radii (comma separated): ", Arrays.toString(radii));
		String[] strings = array.replaceAll("\\s*", "").replace("]", "").replace("[", "").split(",");
		radii = new double[strings.length];
		for (int i = 0; i < strings.length; i++) {
			radii[i]=Double.parseDouble(strings[i]);
		}
		alpha = UserUtil.queryDouble("Alpha value (controls symmetry): ", alpha);
		smallGradientThreshold = UserUtil.queryDouble("Small gradient Threshold [0,1]: ", smallGradientThreshold);
		configured = true;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		return "@article{Loy2003," +
				"abstract = {A new transform is presented that utilizes local radial symmetry to highlight points of interest within a scene. Its low-computational complexity and fast runtimes makes this method well-suited for real-time vision applications. The performance of the transform is demonstrated on a wide variety of images and compared with leading techniques from the literature. Both as a facial feature detector and as a generic region of interest detector the new transform is seen to offer equal or superior performance to contemporary techniques at a relatively low-computational cost. A real-time implementation of the transform is presented running at over 60 frames per second on a standard Pentium III PC.}," +
				"author = {Loy, G. and Zelinsky, A.}," +
				"doi = {10.1109/TPAMI.2003.1217601}," +
				"issn = {0162-8828}," +
				"journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}," +
				"keywords = {Computer vision,Costs,Detectors,Eyes,Face detection,Facial features,Humans,Layout,Pentium III PC,Psychology,Runtime,computational complexity,facial feature detector,fast radial symmetry,feature extraction,local radial symmetry,performance,point of interest detection,real-time systems,real-time vision applications,transform,transforms}," +
				"month = Aug," +
				"number = {8}," +
				"pages = {959--973}," +
				"title = {{Fast radial symmetry for detecting points of interest}}," +
				"volume = {25}," +
				"year = {2003}" +
				"}";
	}

	@Override
	public String getMedlineCitation() {
		return "Loy, G., & Zelinsky, A. (2003). Fast radial symmetry for detecting points of interest. IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(8), 959�973.";
	}


	public static void main(String[] args) {
		CONRAD.setup();
		Opener op = new Opener();
		ImagePlus ip = op.openImage("C:\\Users\\Martin\\Documents\\MATLAB\\FastRadialSymmetry_PointPicker\\bla.tif");
		ip.show();
		FastRadialSymmetryTool frst = new FastRadialSymmetryTool();
		frst.setRadii(new double[]{3});
		frst.setAlpha(3);
		frst.setSmallGradientThreshold(0.02);
		frst.setConfigured(true);
		frst.setImageIndex(0);
		frst.applyToolToImage(ImageUtil.wrapImageProcessor(ip.getStack().getProcessor(1))).show();
	}

}
/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
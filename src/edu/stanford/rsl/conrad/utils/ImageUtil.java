/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.utils;

//TODO: Use our own matrices instead of Jama.Matrix

import java.awt.Frame;
import java.awt.image.IndexColorModel;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.Grid;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.Grid4D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.io.ImagePlusProjectionDataSource;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageWindow;
import ij.measure.Calibration;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;


public abstract class ImageUtil {

	public static void saveAs(Grid3D grid, String string) {
		ImagePlus imp = ImageUtil.wrapGrid3D(grid, string);
		IJ.save(imp, string);
	}
	
	/**
	 * Method to display 3DGrids properly. This method is also able to handle multi channel data.
	 * @param grid the grid
	 * @param title the title
	 * @return the ImagePlus
	 */
	public static ImagePlus wrapGrid3D(Grid3D grid, String title){
		if (grid != null) {
			ImageStack stack;
			if (grid.getSubGrid(0) instanceof Grid2DComplex)
				stack = new ImageStack(grid.getSize()[0]*2, grid.getSize()[1], grid.getSize()[2]);
			else
				stack = new ImageStack(grid.getSize()[0], grid.getSize()[1], grid.getSize()[2]);
			
			if (grid.getSubGrid(0) instanceof MultiChannelGrid2D){
				MultiChannelGrid2D first = (MultiChannelGrid2D) grid.getSubGrid(0);
				String [] names = first.getChannelNames();
				// finalize the hyperstack
				ImagePlus hyper = new ImagePlus();
				ImageStack hyperStack = new ImageStack(grid.getSize()[0], grid.getSize()[1]);
				int dimz = grid.getSize()[2];
				for (int c=0;c < first.getNumberOfChannels(); c++){
					for (int i=0; i< dimz; i++){
						MultiChannelGrid2D current = (MultiChannelGrid2D) grid.getSubGrid(i);
						String frameTitle = "Slice z = " +(i-1);
						if (names != null){
							frameTitle += " Channel: " + names[c]; 
						} else {
							frameTitle += " Channel: " + c;
						}
						hyperStack.addSlice(frameTitle, ImageUtil.wrapGrid2D(current.getChannel(c)));
					}
				}
				setCalibrationToImagePlus(hyper);
				hyper.setStack(title, hyperStack);
				hyper.setDimensions(1, dimz, first.getNumberOfChannels());
				hyper.setOpenAsHyperStack(true);
				return hyper;
			} 
			else {
				for (int i=0; i< stack.getSize(); i++){
					stack.setPixels(grid.getSubGrid(i).getBuffer(), i+1);
				}
				ImagePlus imagePlus = new ImagePlus(title, stack);
				setCalibrationToImagePlus(imagePlus, grid);
				return imagePlus;
			}

		} else 
			return null;
	}
	public static ImagePlus wrapGrid4D(Grid4D grid, String title){
		if (grid != null) {
			ImageStack stack = new ImageStack(grid.getSize()[0], grid.getSize()[1], grid.getSize()[2]);
			if (grid.getSubGrid(0) instanceof MultiChannelGrid3D){
				MultiChannelGrid3D first = (MultiChannelGrid3D) grid.getSubGrid(0);
				String [] names = first.getChannelNames();
				// finalize the hyperstack
				ImagePlus hyper = new ImagePlus();
				ImageStack hyperStack = new ImageStack(grid.getSize()[0], grid.getSize()[1]);
				int dimz = grid.getSize()[2];
				int dimy=  grid.getSize()[1];
				for (int c=0;c < first.getNumberOfChannels(); c++){
					for (int i=0; i< dimz; i++){
						for(int j=0; j<dimy; j++){
						MultiChannelGrid3D current = (MultiChannelGrid3D) grid.getSubGrid(i);
						String frameTitle = "Slice z = " +(i-1);
						if (names != null){
							frameTitle += " Channel: " + names[c]; 
						} else {
							frameTitle += " Channel: " + c;
						}
						hyperStack.addSlice(frameTitle, ImageUtil.wrapGrid2D(current.getChannel(c).getSubGrid(j)));
					}
				}
				}
				setCalibrationToImagePlus(hyper);
				hyper.setStack(title, hyperStack);
				hyper.setDimensions(1, dimz, first.getNumberOfChannels());
				hyper.setOpenAsHyperStack(true);
				return hyper;
			} else {
				for (int i=0; i< stack.getSize(); i++){
					stack.setPixels(grid.getSubGrid(i).getBuffer(), i+1);
				}
				ImagePlus imagePlus = new ImagePlus(title, stack);
				setCalibrationToImagePlus(imagePlus, grid);
				return imagePlus;
			}

		} else 
			return null;
	}
	private static void setCalibrationToImagePlus(ImagePlus image){
		Calibration calibration = image.getCalibration();
		calibration.xOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		calibration.yOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		calibration.zOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
		calibration.pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingX();
		calibration.pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingY();
		calibration.pixelDepth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();
	}
	
	private static void setCalibrationToImagePlus2D(ImagePlus image){
		Calibration calibration = image.getCalibration();
		calibration.xOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		calibration.yOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		calibration.pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingX();
		calibration.pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingY();
	}

	private static void setCalibrationToImagePlus(ImagePlus imagePlus, Grid grid){
		Calibration calibration = imagePlus.getCalibration();
		calibration.xOrigin = General.worldToVoxel(-grid.getOrigin()[0], grid.getSpacing()[0], 0);
		calibration.yOrigin = General.worldToVoxel(-grid.getOrigin()[1], grid.getSpacing()[1], 0);
		calibration.zOrigin = General.worldToVoxel(-grid.getOrigin()[2], grid.getSpacing()[2], 0);
		calibration.pixelWidth = grid.getSpacing()[0];
		calibration.pixelHeight = grid.getSpacing()[1];
		calibration.pixelDepth = grid.getSpacing()[2];
	}

	private static void setCalibrationToImagePlus2D(ImagePlus imagePlus, Grid grid){
		Calibration calibration = imagePlus.getCalibration();
		calibration.xOrigin = General.worldToVoxel(-grid.getOrigin()[0], grid.getSpacing()[0], 0);
		calibration.yOrigin = General.worldToVoxel(-grid.getOrigin()[1], grid.getSpacing()[1], 0);
		calibration.pixelWidth = grid.getSpacing()[0];
		calibration.pixelHeight = grid.getSpacing()[1];
	}

	/**
	 * Method to display 3DGrids properly. This method is also able to handle multi channel data.
	 * @param grid the grid
	 * @param title the title
	 * @return the ImagePlus
	 */
	public static ImagePlus wrapGrid(NumericGrid grid, String title){
		if (grid != null) {
			if (grid instanceof Grid3D) return wrapGrid3D((Grid3D) grid, title);
			else if (grid instanceof Grid2D){
				if (grid instanceof MultiChannelGrid2D){
					MultiChannelGrid2D first = (MultiChannelGrid2D) grid;
					String [] names = first.getChannelNames();
					// finalize the hyperstack
					ImagePlus hyper = new ImagePlus();
					ImageStack hyperStack = new ImageStack(grid.getSize()[0], grid.getSize()[1]);
					for (int c=0;c < first.getNumberOfChannels(); c++){
						String frameTitle = "";
						if (names != null){
							frameTitle += "Channel: " + names[c]; 
						} else {
							frameTitle += "Channel: " + c;
						}
						hyperStack.addSlice(frameTitle, ImageUtil.wrapGrid2D(first.getChannel(c)));
					}
					setCalibrationToImagePlus2D(hyper);
					hyper.setStack(title, hyperStack);
					hyper.setDimensions(1, 1, first.getNumberOfChannels());
					hyper.setOpenAsHyperStack(true);
					return hyper;
				} else {
					FloatProcessor proc = wrapGrid2D((Grid2D) grid);
					ImagePlus iPlus = new ImagePlus(title, proc);
					setCalibrationToImagePlus2D(iPlus, grid);
					return iPlus;
				}
			}
		} 
		return null;
	}


	public static FloatProcessor wrapGrid2D(Grid2D grid){
		return new FloatProcessor(grid.getWidth(), grid.getHeight(), grid.getBuffer(), null);
	}

	public static Grid3D wrapImagePlus(ImagePlus image){
		return wrapImagePlus(image, false, false);
	}

	public static Grid3D wrapImagePlus(ImagePlus image, boolean copy){
		return wrapImagePlus(image, copy, false);
	}

	public static Grid3D wrapImagePlus(ImagePlus image, boolean copy, boolean invertStack){
		int dimz = image.getDimensions()[3];
		Grid3D revan = new Grid3D(image.getWidth(), image.getHeight(), dimz, false);
		if (image.isHyperStack()){
			if (invertStack) throw new RuntimeException ("Stack inversion is not implemented for hyperstacks yet. Sorry");
			int channels = image.getDimensions()[4];
			String [] channelNames= new String[channels];
			for (int c=0; c<channels;c++){
				if (image.getImageStack().getSliceLabel((c*dimz)+1) != null){
					String [] splitStrings =image.getImageStack().getSliceLabel((c*dimz)+1).split("Channel: "); 
					if (splitStrings != null){
						if (splitStrings.length > 1){
							channelNames[c]=splitStrings[1];
						} else {
							channelNames[c]=splitStrings[0];
						}
					} else {
						channelNames[c]="Channel " + c;
					}
				} else {
					channelNames[c]="Channel " +c;
				}
			}
			for (int i = 0; i < dimz; i++){
				MultiChannelGrid2D multiChannelGrid2D = new MultiChannelGrid2D(image.getWidth(), image.getHeight(), channels);
				multiChannelGrid2D.setChannelNames(channelNames);
				for (int c= 0; c<channels;c++){
					Grid2D grid2d = wrapImagePlusSlice(image, (i+(dimz*c))+1, copy);
					multiChannelGrid2D.setChannel(c, grid2d);
				}
				revan.setSubGrid(i, multiChannelGrid2D);
			}
		} else {
			for (int i = 0; i < image.getDimensions()[3]; i++){
				int pos = i;
				if (invertStack) pos = image.getDimensions()[3] - 1 - i;
				revan.setSubGrid(pos, wrapImagePlusSlice(image, i+1, copy));
			}
		}
		revan.setSpacing(image.getCalibration().pixelWidth, image.getCalibration().pixelHeight, image.getCalibration().pixelDepth);
		double xorigin = General.voxelToWorld(-image.getCalibration().xOrigin, revan.getSpacing()[0], 0);
		double yorigin = General.voxelToWorld(-image.getCalibration().yOrigin, revan.getSpacing()[1], 0);
		double zorigin = General.voxelToWorld(-image.getCalibration().zOrigin, revan.getSpacing()[2], 0);
		//System.out.println(image.getCalibration().pixelWidth  + " " + image.getCalibration().pixelHeight + " " + image.getCalibration().pixelDepth);
		revan.setOrigin(xorigin, yorigin, zorigin);
		return revan;
	}

	public static void applyConradImageCalibration(ImagePlus image, boolean isReconstruction){
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		// FIXME: If we have a projection domain output the x and y spacing and origin refer to the detector
		// In case of a reconstruction output this dimensions should be equal to the global settings!
		// FIXME: Find a nicer way to handle difference between projection and reconstructed domain!!
		Calibration cal = image.getCalibration();
		if (isReconstruction){
			cal.setXUnit("mm");
			cal.setYUnit("mm");
			cal.setZUnit("mm");
			cal.xOrigin = traj.getOriginInPixelsX();
			cal.yOrigin = traj.getOriginInPixelsY();
			cal.zOrigin = traj.getOriginInPixelsZ();
			cal.pixelWidth = traj.getVoxelSpacingX();
			cal.pixelHeight = traj.getVoxelSpacingY();
			cal.pixelDepth = traj.getVoxelSpacingZ();
		}
		else
		{
			cal.setXUnit("mm");
			cal.setYUnit("mm");
			cal.setZUnit("degree");
			cal.xOrigin = traj.getDetectorOffsetU();
			cal.yOrigin = traj.getDetectorOffsetV();
			cal.zOrigin = 0;//traj.getPrimaryAngles()[0]/traj.getAverageAngularIncrement();
			cal.pixelWidth = traj.getPixelDimensionX();
			cal.pixelHeight = traj.getPixelDimensionY();
			cal.pixelDepth = traj.getAverageAngularIncrement();
		}
	}

	public static Grid2D wrapImagePlusSlice(ImagePlus image, int n, boolean copy){
		ImageProcessor ip = image.getStack().getProcessor(n);
		return wrapImageProcessor(ip, copy);
	}

	public static Grid2D wrapImageProcessor(ImageProcessor ip){
		return wrapImageProcessor(ip, false);
	}

	public static Grid2D wrapImageProcessor(ImageProcessor ip, boolean copy){
		if (ip instanceof FloatProcessor){
			if(!copy)
				return wrapFloatProcessor((FloatProcessor) ip);
			else return wrapFloatProcessor((FloatProcessor) ip.duplicate());
		} else {
			return wrapFloatProcessor((FloatProcessor) ip.toFloat(0, null));
		}
	}

	public static Grid2D wrapFloatProcessor(FloatProcessor fl){
		return new Grid2D((float[])fl.getPixels(), fl.getWidth(), fl.getHeight());
	}

	public static IndexColorModel getDefaultColorModel(){
		byte[] r = new byte[256];
		byte[] g = new byte[256];
		byte[] b = new byte[256];
		for(int i=0; i<256; i++) {
			r[i]=(byte)i;
			g[i]=(byte)i;
			b[i]=(byte)i;
		}
		return new IndexColorModel(8, 256, r, g, b);
	}

	public static float [] estimateConvolutionKernel(FloatProcessor before, FloatProcessor after, int kernelSize, int number){
		//float [] revan = new float [kernelSize * kernelSize];
		Jama.Matrix observationsAfter = new Jama.Matrix(1, number);
		Jama.Matrix observationsBefore = new Jama.Matrix(kernelSize * kernelSize, number);
		java.util.Random random = new java.util.Random();
		int offset = (-1) * kernelSize / 2;
		// Fill arrays with randomly picked values from the image processors
		for (int n = 0; n < number; n++){
			int x = kernelSize + (int) (random.nextDouble() * (before.getWidth() - (2 * kernelSize)) );
			int y = kernelSize + (int) (random.nextDouble() * (before.getHeight() - (2 * kernelSize)) );
			if (after.getPixelValue(x, y) != 0){
				observationsAfter.set(0, n, after.getPixelValue(x, y));
				int count = 0;
				for (int i=0;i<kernelSize;i++){
					for (int j=0;j<kernelSize;j++){
						observationsBefore.set(count, n, before.getPixelValue(offset + x + i, offset + y + j));
						count++;
					}
				}
			} else {
				// redo this point;
				n--;
			}
		}
		//observationsBefore.print(8, 3);
		Jama.SingularValueDecomposition svd = observationsBefore.transpose().svd();

		//System.out.println(observationsBefore.rank());
		Jama.Matrix sigma = svd.getS().inverse();
		Jama.Matrix inverse = svd.getV().times(sigma).times(svd.getU().transpose()).transpose();
		Jama.Matrix kernelEstimate = observationsAfter.times(inverse);
		int count =0;
		for (int i=0;i<kernelSize;i++){
			for (int j=0;j<kernelSize;j++){
				if(kernelEstimate.get(0, count) < 0) kernelEstimate.set(0, count, kernelEstimate.get(0, count) * -1) ;
				count ++;
			}
			System.out.println();
		}	
		return ij.util.Tools.toFloat(kernelEstimate.getColumnPackedCopy());
	}


	public static ArrayList<ImagePlus> getAvailableImagePlus(){
		ArrayList<ImagePlus> list = new ArrayList<ImagePlus>();
		Frame [] frames = ImageJ.getFrames();
		for (Frame frame: frames){
			if (frame instanceof ImageWindow){
				ImageWindow window = (ImageWindow)frame;
				if (! window.isClosed()){
					list.add(window.getImagePlus());
				}
			}
		}
		return list;
	}

	public static ImagePlus [] getAvailableImagePlusAsArray(){
		ArrayList<ImagePlus> list = getAvailableImagePlus();
		ImagePlus [] array = new ImagePlus[list.size()];
		for(int i=0; i< list.size(); i++){
			array[i] = list.get(i);
		}
		return array;
	}


	/**
	 * Determines the minimal value of a given ImagePlus.
	 * 
	 * @param image the ImagePlus
	 * @return the minimal value
	 */
	public static double minOfImagePlusValues(ImagePlus image){
		if (image.getStackSize() == 1){
			return minOfImageProcessor(image.getChannelProcessor());
		} else {
			double [] mins = new double[image.getStackSize()];
			for (int i = 0; i < image.getStackSize(); i++){
				mins[i] = minOfImageProcessor(image.getStack().getProcessor(i+1));
			}
			return DoubleArrayUtil.minOfArray(mins);
		}
	}

	/**
	 * returns the minimal value of a given ImageProcessor
	 * @param imp the ImageProcessor
	 * @return the minimal value
	 */
	public static double minOfImageProcessor(ImageProcessor imp){
		double min = Double.MAX_VALUE;
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				if (imp.getPixelValue(i, j) < min) {
					min = imp.getPixelValue(i, j);
				}
			}
		}
		return min;
	}

	/**
	 * returns the minimal and the maxiaml value of a given ImageProcessor
	 * @param imp the ImageProcessor
	 * @return the minimal and the maximal value as double array
	 */
	public static double [] minAndMaxOfImageProcessor(ImageProcessor imp){
		double [] revan = new double [2];
		revan[0] = Double.MAX_VALUE;
		revan[1] = -Double.MAX_VALUE;
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				if (imp.getPixelValue(i, j) < revan[0]) {
					revan[0] = imp.getPixelValue(i, j);
				}
				if (imp.getPixelValue(i, j) > revan[1]) {
					revan[1] = imp.getPixelValue(i, j);
				}
			}

		}
		return revan;
	}

	/**
	 * returns the minimal and the maxiaml value of a given ImageProcessor
	 * @param image the ImageProcessor
	 * @return the minimal and the maximal value as double array
	 */
	public static double [] minAndMaxOfImageProcessor(ImagePlus image){
		double [] revan = new double [2];
		revan[0] = Double.MAX_VALUE;
		revan[1] = Double.MIN_VALUE;
		for (int k = 1; k <= image.getStackSize(); k++){
			ImageProcessor imp = image.getStack().getProcessor(k);
			for (int i = 0; i < imp.getWidth(); i++){
				for (int j = 0; j < imp.getHeight(); j++){
					if (imp.getPixelValue(i, j) < revan[0]) {
						revan[0] = imp.getPixelValue(i, j);
					}
					if (imp.getPixelValue(i, j) > revan[1]) {
						revan[1] = imp.getPixelValue(i, j);
					}
				}

			}
		}
		return revan;
	}

	/**
	 * Increases the pixel values of all pixels in the ImagePlus by addition of value
	 * 
	 * @param image the ImagePlus
	 * @param value the value to add
	 */
	public static void addToImagePlusValues(ImagePlus image, double value){
		if (image.getStackSize() == 1){
			image.getChannelProcessor().add(value);
		} else {
			for (int i = 0; i < image.getStackSize(); i++){
				image.getStack().getProcessor(i+1).add(value);
			}
		}
	}

	/**
	 * Adds the pixel values of all pixels in the ImagePlus by  the second Image
	 * 
	 * @param image the ImagePlus
	 * @param image2 the values to add to
	 */
	public static void addImagePlusValues(ImagePlus image, ImagePlus image2){
		if (image.getStackSize() == 1){
			addProcessors(image.getChannelProcessor(), image2.getChannelProcessor());
		} else {
			for (int i = 0; i < image.getStackSize(); i++){
				addProcessors(image.getStack().getProcessor(i), image2.getStack().getProcessor(i));
			}
		}
	}

	/**
	 * Multiplies the pixel values of all pixels in the ImagePlus by  value
	 * 
	 * @param image the ImagePlus
	 * @param value the value to multiply
	 */
	public static void multiplyImagePlusValues(ImagePlus image, double value){
		if (image.getStackSize() == 1){
			image.getChannelProcessor().multiply(value);
		} else {
			for (int i = 0; i < image.getStackSize(); i++){
				image.getStack().getProcessor(i+1).multiply(value);
			}
		}
	}

	/**
	 * Normalizes all pixel values of an ImagePlus to mean 0 and standard deviation 1.
	 * 
	 * @param image the imagePlus
	 * @return an array with two entries. The first one is the mean of all pixel values. The second one the standard deviation of all pixel values.
	 */
	public static double [] normalizeImagePlusCutOff(ImagePlus image, int numStandardDeviations){
		if (image.getStackSize() == 1){
			return ImageUtil.normalizeImageProcessorCutOff(image.getChannelProcessor(), numStandardDeviations);
		} else {
			double [] revan = new double [2];
			double [] means = new double [image.getStackSize()];
			double [] stddevs = new double [image.getStackSize()];
			for (int i = 0; i < image.getStackSize(); i++){
				double [] temp = ImageUtil.normalizeImageProcessorCutOff(image.getStack().getProcessor(i+1), numStandardDeviations);
				means[i] = temp[0];
				stddevs[i] = temp[1];
			}
			revan[0] = DoubleArrayUtil.computeMean(means);
			revan[1] = DoubleArrayUtil.computeMean(stddevs);
			return revan;
		}
	}

	/**
	 * Normalizes all pixel values of an ImagePlus to mean 0 and standard deviation 1.
	 * 
	 * @param image the imagePlus
	 * @return an array with two times stack size entries. The respective first one is the mean of all pixel values. The second one the standard deviation of all pixel values.
	 */
	public static double [] normalizeImagePlus(ImagePlus image){
		if (image.getStackSize() == 1){
			return ImageUtil.normalizeImageProcessor(image.getChannelProcessor());
		} else {
			double [] revan = new double [image.getStackSize() * 2];
			for (int i = 0; i < image.getStackSize(); i++){
				double [] temp = ImageUtil.normalizeImageProcessor(image.getStack().getProcessor(i+1));
				revan[2 * i] = temp[0];
				revan[(2 * i) + 1] = temp[1];
			}
			return revan;
		}
	}

	/**
	 * Normalizes all pixel values of an ImagePlus to minimum 0 and maximum 1.
	 * 
	 * @param image the imagePlus
	 * @return an array with two entries. The first one is the mean of all pixel values. The second one the standard deviation of all pixel values.
	 */
	public static double [] normalizeImagePlusMinMax(ImagePlus image){
		if (image.getStackSize() == 1){
			return ImageUtil.normalizeImageProcessorMinMax(image.getChannelProcessor());
		} else {
			double [] revan = new double [2];
			double [] means = new double [image.getStackSize()];
			double [] stddevs = new double [image.getStackSize()];
			for (int i = 0; i < image.getStackSize(); i++){
				double [] temp = ImageUtil.normalizeImageProcessorMinMax(image.getStack().getProcessor(i+1));
				means[i] = temp[0];
				stddevs[i] = temp[1];
			}
			revan[0] = DoubleArrayUtil.computeMean(means);
			revan[1] = DoubleArrayUtil.computeMean(stddevs);
			return revan;
		}
	}

	/** Normalizes an ImageProcessor to minimum 0 and maximum 1
	 * 
	 * @param imp the image processor
	 * @return an double array with two entries. The first one is the mean and the second one is the standard deviation.
	 */
	public static double [] normalizeImageProcessorMinMax(ImageProcessor imp){
		double [] revan = ImageUtil.minAndMaxOfImageProcessor(imp);
		imp.add((-1) * revan[0]);
		double range = revan[1] - revan[0];
		imp.multiply(1 / range);
		//CONRAD.log("Min: " + revan[0] + " Max: " + revan[1]);
		return revan;
	}

	/** Normalizes an ImageProcessor to mean 0 and standard deviation 1
	 * 
	 * @param imp the image processor
	 * @return an double array with two entries. The first one is the mean and the second one is the standard deviation.
	 */
	public static double [] normalizeImageProcessor(ImageProcessor imp){
		double [] revan = new double [2];
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				revan[0] += imp.getPixelValue(i, j);
			}
		}
		revan[0] /= (imp.getWidth() * imp.getHeight());
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				double value = imp.getPixelValue(i, j) - revan[0];
				imp.putPixelValue(i, j, value);
				revan[1] += Math.pow(value, 2);
			}
		}
		revan[1] /= imp.getWidth() * imp.getHeight();
		revan[1] = Math.sqrt(revan[1]);
		for (int i=0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				double value = imp.getPixelValue(i, j) / revan[1];
				imp.putPixelValue(i, j, value);
			}
		}
		//CONRAD.log("Mean: " + revan[0] + " Standard deviation: " + revan[1]);
		return revan;
	}

	/** Normalizes an ImageProcessor to mean 0 and standard deviation 1. Cuts off values greater than a certain amount of standard deviations.
	 * 
	 * @param imp the image processor
	 * @param numStandardDeviations Number of standard deviations after which the values are cut off
	 * @return an double array with two entries. The first one is the mean and the second one is the standard deviation.
	 */
	public static double [] normalizeImageProcessorCutOff(ImageProcessor imp, int numStandardDeviations){
		double [] revan = new double [2];
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				revan[0] += imp.getPixelValue(i, j);
			}
		}
		revan[0] /= (imp.getWidth() * imp.getHeight());
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				double value = imp.getPixelValue(i, j) - revan[0];
				imp.putPixelValue(i, j, value);
				revan[1] += Math.pow(value, 2);
			}
		}
		revan[1] /= imp.getWidth() * imp.getHeight();
		revan[1] = Math.sqrt(revan[1]);
		for (int i=0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				double value = imp.getPixelValue(i, j) / revan[1];
				if (value > numStandardDeviations){
					value = numStandardDeviations;
				}
				if (value < numStandardDeviations * (-1)){
					value = - numStandardDeviations;
				}
				imp.putPixelValue(i, j, value);
			}
		}
		//CONRAD.log("Mean: " + revan[0] + " Standard deviation: " + revan[1]);
		return revan;
	}

	/** Normalizes an ImageProcessor to mean 0 and standard deviation 1
	 * 
	 * @param imp the image processor
	 * @return an double array with two entries. The first one is the mean and the second one is the standard deviation.
	 */
	public static double [] computeMeanAndStandardDeviation(ImageProcessor imp){
		double [] revan = new double [2];
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				revan[0] += imp.getPixelValue(i, j);
			}
		}
		revan[0] /= imp.getWidth() * imp.getHeight();
		for (int i = 0; i < imp.getWidth(); i++){
			for (int j = 0; j < imp.getHeight(); j++){
				revan[1] += Math.pow(imp.getPixelValue(i, j) - revan[0], 2);
			}
		}
		revan[1] /= imp.getWidth() * imp.getHeight();
		revan[1] = Math.sqrt(revan[1]);
		return revan;
	}

	public static FloatProcessor divideImages(ImagePlus nominator, ImagePlus denominator){
		FloatProcessor information = (FloatProcessor) nominator.getChannelProcessor().duplicate();
		for (int i = 0; i < information.getWidth(); i++) {
			for (int j = 0; j < information.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (nominator.getChannelProcessor().getPixelValue(i, j) != 0) {
					value = nominator.getChannelProcessor().getPixelValue(i, j) / denominator.getChannelProcessor().getPixelValue(i, j);	
				} 
				information.putPixelValue(i, j, value);
			}
		}
		return information;
	}

	public static FloatProcessor divideImages(ImageProcessor nominator, ImageProcessor denominator){
		FloatProcessor information = (FloatProcessor) nominator.duplicate();
		for (int i = 0; i < information.getWidth(); i++) {
			for (int j = 0; j < information.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (nominator.getPixelValue(i, j) != 0) {
					value = nominator.getPixelValue(i, j) / denominator.getPixelValue(i, j);	
				} 
				information.putPixelValue(i, j, value);
			}
		}
		return information;
	}

	public static FloatProcessor multiplyImages(ImagePlus nominator, ImagePlus denominator, int n){
		FloatProcessor information = (FloatProcessor) denominator.getStack().getProcessor(n).duplicate();
		for (int i = 0; i < information.getWidth(); i++) {
			for (int j = 0; j < information.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (nominator.getChannelProcessor().getPixelValue(i, j) != Float.NaN) {
					value = nominator.getStack().getProcessor(n).getPixelValue(i, j) * denominator.getStack().getProcessor(n).getPixelValue(i, j);	
				} 
				information.putPixelValue(i, j, value);
			}
		}
		return information;
	}

	public static FloatProcessor multiplyImages(ImageProcessor nominator, ImageProcessor denominator){
		FloatProcessor information = denominator.duplicate().toFloat(0, null);
		for (int i = 0; i < information.getWidth(); i++) {
			for (int j = 0; j < information.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (nominator.getPixelValue(i, j) != Float.NaN) {
					value = nominator.getPixelValue(i, j) * denominator.getPixelValue(i, j);	
				} 
				information.putPixelValue(i, j, value);
			}
		}
		return information;
	}

	/**
	 * Divides two image processors in the two given ImagePlus. Will allocate a new FloatProcessor for the result.
	 * 
	 * @param nominator the nominator
	 * @param denominator the denominator
	 * @param n the index of the stack (starts with 0).
	 * @return the division result.
	 */
	public static FloatProcessor divideImages(ImagePlus nominator, ImagePlus denominator, int n){
		FloatProcessor information = (FloatProcessor) nominator.getStack().getProcessor(n+1).duplicate();
		for (int i = 0; i < information.getWidth(); i++) {
			for (int j = 0; j < information.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (nominator.getChannelProcessor().getPixelValue(i, j) != 0) {
					value = nominator.getStack().getProcessor(n+1).getPixelValue(i, j) / denominator.getStack().getProcessor(n+1).getPixelValue(i, j);	
				} 
				information.putPixelValue(i, j, value);
			}
		}
		return information;
	}


	/**
	 * Adds two image processors. Works in place. First ImageProcessor is modified.
	 * 
	 * @param left the first processor
	 * @param right the second processor
	 */
	public static void addProcessors(ImageProcessor left, ImageProcessor right){
		for (int i = 0; i < left.getWidth(); i++) {
			for (int j = 0; j < left.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (left.getPixelValue(i, j) != Float.NaN) {
					value = left.getPixelValue(i, j) + right.getPixelValue(i, j);	
				} 
				left.putPixelValue(i, j, value);
			}
		}
	}

	/**
	 * Subtracts two image processors. Works in place. First ImageProcessor is modified.
	 * 
	 * @param left the first processor
	 * @param right the second processor
	 */
	public static void subtractProcessors(ImageProcessor left, ImageProcessor right){
		for (int i = 0; i < left.getWidth(); i++) {
			for (int j = 0; j < left.getHeight(); j++) {
				double value = Double.MAX_VALUE;
				if (left.getPixelValue(i, j) != Float.NaN) {
					value = left.getPixelValue(i, j) - right.getPixelValue(i, j);	
				} 
				left.putPixelValue(i, j, value);
			}
		}
	}

	/**
	 * Creates a unigue String representation of an array of ImagePlus
	 * @param images
	 * @return the String []
	 */
	public static String[] getStringArrayRepresentation(ImagePlus[] images) {
		String [] names = new String[images.length];
		for (int i= 0; i < names.length; i++){
			names[i]= images[i].toString() + " (Image " + i + ")"; 
		}
		return names;
	}

	/**
	 * Returns the matching image given it's String representation
	 * @param name the String representation of the image
	 * @param images the array of ImagePlus
	 * @return the ImagePlus
	 */
	public static ImagePlus getImagePlusFromString(String name, ImagePlus [] images){
		ImagePlus selection = null;
		for (int i= 0; i < images.length; i++){
			if (name.equals(images[i].toString() + " (Image " + i + ")")){
				selection = images[i];
				break;
			}
		}
		return selection;
	}

	/**
	 * Creates a two-dimensional symmetric gaussian filter kernel equal to the Matlab method
	 * @param sizeX The kernel x size
	 * @param sizeY The kernel y size
	 * @param sigma The kernel's standard deviation value
	 * @return the grid2d
	 */
	public static Grid2D create2DGauss(int sizeX, int sizeY, double sigma){
		Grid2D out = new Grid2D(sizeX,sizeY);
		//double maxEps = Double.MIN_VALUE;
		float sum = 0;
		for (int y = 0; y < out.getHeight(); y++) {
			for (int x = 0; x < out.getWidth(); x++) {
				double xx = x - (double)out.getWidth()/2.0 + 0.5;
				double yy = y - (double)out.getHeight()/2.0 + 0.5;
				//out.setAtIndex(x, y, (float)(((double)size) * Math.exp(-0.5*(xx*xx+yy*yy)/sigma/sigma)/(2*Math.PI*Math.sqrt(2)*sigma)));
				float val = (float)Math.exp(-(xx*xx+yy*yy)/2/sigma/sigma);
				out.setAtIndex(x, y, val);
				sum += val;
				//if (val > maxEps)
				//	maxEps = val;
			}
		}

		// set very small values to 0 and normalize kernel to sum 1
		/*		maxEps *= 1.1921e-07;
		for (int y = 0; y < out.getHeight(); y++) {
			for (int x = 0; x < out.getWidth(); x++) {
				if (out.getAtIndex(x, y) > maxEps)
					out.setAtIndex(x, y, out.getAtIndex(x, y)/sum);
				else
					out.setAtIndex(x,y,0);
			}
		}*/

		NumericPointwiseOperators.divideBy(out, sum);

		return out;
	}

	/**
	 * 
	 * @param inputStack the input stack
	 * @param filter the tool to apply
	 * @return Grid3D the output of the parallel processing
	 */
	public static Grid3D applyFilterInParallel(Grid3D inputStack, ImageFilteringTool filter){
		// run all the filters in parallel on the slices
		return applyFilterInParallel(inputStack, filter, false);

	}

	/**
	 * 
	 * @param inputStack the input stack
	 * @param filter the tool to apply
	 * @return Grid3D the output of the parallel processing
	 */
	public static Grid3D applyFilterInParallel(Grid3D inputStack, ImageFilteringTool filter, boolean showStatus){
		// run all the filters in parallel on the slices
		return applyFiltersInParallel(inputStack, new ImageFilteringTool[] {filter}, showStatus);

	}

	/**
	 * 
	 * @param inputStack
	 * @param filters
	 * @return Grid3D the output of the parallel processing
	 */
	public static Grid3D applyFiltersInParallel(Grid3D inputStack, ImageFilteringTool[] filters){
		return applyFiltersInParallel(inputStack, filters, false);
	}

	/**
	 * 
	 * @param inputStack
	 * @param filters
	 * @return Grid3D the output of the parallel processing
	 */
	public static Grid3D applyFiltersInParallel(Grid3D inputStack, ImageFilteringTool[] filters, boolean showStatus){
		// run all the filters in parallel on the slices
		Grid3D outputStack=null;
		try {
			ImagePlusDataSink sink = new ImagePlusDataSink();
			sink.configure();
			ImagePlusProjectionDataSource pSource = new ImagePlusProjectionDataSource();
			pSource.setImage(inputStack);
			ParallelImageFilterPipeliner filteringPipeline = new ParallelImageFilterPipeliner(pSource, filters, sink);
			filteringPipeline.project(showStatus);
			outputStack = sink.getResult();
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return outputStack;

	}

}

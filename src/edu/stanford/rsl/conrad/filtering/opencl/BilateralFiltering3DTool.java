/*
 * Copyright (C) 2018 Benedikt Lorch, Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.opencl;

import ij.IJ;
import ij.ImagePlus;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.FloatBuffer;

import math3d.Point3d;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLMemory;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Tool for computation of the Bilateral Filter 3-D in OpenCL.
 * This tool uses the CONRAD internal Grid 3-D data structure.
 * @author Benedikt Lorch
 *
 */
public class BilateralFiltering3DTool extends OpenCLFilteringTool3D {
	
	private static final long serialVersionUID = 8268932618986579399L;
	protected double sigmaPhoto = 1;
	protected double[] sigmaGeom = { 5.d, 5.d, 5.d };
	
	public static final double KERNEL_MIN_WEIGHT = 0.001;	// Used to make a suggestion for the kernelWidth depending on sigmaGeom
	protected int kernelWidth;
	protected CLBuffer<FloatBuffer> geometricKernel;
	
	protected boolean askForGuidance = true;
	protected boolean showGuidance = false; // Use Joint Bilateral Filter
	protected Grid3D guidanceGrid;
	protected CLBuffer<FloatBuffer> template;
	
	
	public BilateralFiltering3DTool() {
		this.kernelName = kernelname.BILATERAL_FILTER_3D;
	}
	
	@Override
	public void configure() throws Exception {
		
		this.kernelName = kernelname.BILATERAL_FILTER_3D;
		
		double[] sGeom = UserUtil.queryArray("Enter geometric sigma (for each dimension)", sigmaGeom);
		if (sGeom.length == 3) {
			// If the given arrays contains three values, use these as dimensions
			this.sigmaGeom = sGeom;
		} else if (sGeom.length == 1) {
			// In case only a single paramater has been entered, use it for each dimension
			this.sigmaGeom = new double[] {sGeom[0], sGeom[0], sGeom[0]};
		} else {
			throw new IllegalArgumentException("Please enter three double values as geometric sigma for each dimension or one double value for all dimensions");
		}
		
		
		this.sigmaPhoto = UserUtil.queryDouble("Enter photometric sigma", sigmaPhoto);
		
		kernelWidth = computeKernelWidth();
		this.kernelWidth = UserUtil.queryInt("Enter kernel width", kernelWidth);
		if (kernelWidth % 2 == 0) {
			throw new IllegalArgumentException("Kernel's width needs to be an odd number.");
		}
		
		if (this.askForGuidance) {
			this.showGuidance = UserUtil.queryBoolean("Add guidance image in first channel?");	// Does the user want to use Joint Bilateral Filter?
			
			// Let the user select the guidance image from his file system
			if (this.showGuidance) {
				String guidanceImageFile = FileUtil.myFileChoose(".tif", false);
				if (null == guidanceImageFile) {
					throw new IllegalArgumentException("The user claimed to add a guidance image but didn't select one upon being prompted.");
				}
				else if (!new File(guidanceImageFile).exists()) {
					throw new FileNotFoundException("The specified guidance image file " + guidanceImageFile + " does not exist."); 
				}
				ImagePlus ip = IJ.openImage(guidanceImageFile);
				this.guidanceGrid = ImageUtil.wrapImagePlus(ip);
			}
		}
		
		configured = true;

	}
	
	/**
	 * Called by process() before the processing begins. Put your write buffers to the queue here.
	 * @param input Grid 3-D to be processed
	 * @param queue CommandQueue for the specific device
	 */
	@SuppressWarnings("unused")
	@Override
	protected void prepareProcessing(Grid3D input, CLCommandQueue queue) {
		
		// Check for errors with guidance grid
		if (showGuidance) {
			if (null == guidanceGrid) {
				throw new IllegalArgumentException("The user claimed to add a guidance image but the guidance image could not be found.");
			}
			else {
				// Assure equal dimensions
				int[] guidanceSize = guidanceGrid.getSize();
				if (width != guidanceSize[0]
						|| height != guidanceSize[1]
						|| depth != guidanceSize[2]) {
					throw new IllegalArgumentException("The given guidence image's dimensions are not equal to the sizes which this filter has been configured for.");
				}
			}
		}
		else if (null != guidanceGrid) {
			// Guidance image given but not used
			String message = "You passed a template image to process() but claimed not to use a guidance image. Your template image will be ignored. In order to use the template image with Joint Bilateral Filter, click 'yes' when prompted whether to use a guidance image.";
			if (debug > 0) {
				System.err.println(message);
			}
		}
		
		// Copy image data into linear floatBuffer
		gridToBuffer(image.getBuffer(), input);
		image.getBuffer().rewind();
		queue.putWriteBuffer(image, true);
		
		// Copy guidance image
		if (showGuidance && null != guidanceGrid) {
			template = clContext.createFloatBuffer(this.width * this.height * this.depth, CLMemory.Mem.READ_ONLY);
			gridToBuffer(template.getBuffer(), guidanceGrid);
			template.getBuffer().rewind();
			queue.putWriteBuffer(template, true);
		}
		
		// Compute geometric kernel: If the kernel is a sphere, then use symmetric properties to speed up processing
		Grid3D geometricKernelGrid = (sigmaGeom[0] == sigmaGeom[1] && sigmaGeom[1] == sigmaGeom[2]) ? computeSymmetricGeometricKernel() : computeGeometricKernel();
		geometricKernel = clContext.createFloatBuffer(kernelWidth * kernelWidth * kernelWidth, CLMemory.Mem.READ_ONLY);
		gridToBuffer(geometricKernel.getBuffer(), geometricKernelGrid);
		geometricKernel.getBuffer().rewind();
		queue.putWriteBuffer(geometricKernel, true);
	}
	
	// Accessors and mutators for configuration variables
	public void setSigmaPhoto(double sigmaPhoto) {
		this.sigmaPhoto = sigmaPhoto;
	}
	
	public void setSigmaGeom(double[] sigmaGeom) {
		this.sigmaGeom = sigmaGeom;
	}
	
	public void setKernelWidth() {
		
		assert(this.sigmaGeom.length == 3);
		assert(this.sigmaGeom[0] != 0.d);
		assert(this.sigmaGeom[1] != 0.d);
		assert(this.sigmaGeom[2] != 0.d);
		assert(this.sigmaPhoto != 0.d);
		
		this.kernelWidth = computeKernelWidth();
	}
	
	public void setKernelWidth(int kernelWidth) {
		this.kernelWidth = kernelWidth;
	}
	
	public double getSigmaPhoto() {
		return this.sigmaPhoto;
	}
	
	public double[] getSigmaGeom() {
		return this.sigmaGeom;
	}
	
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}
	
	
	/**
	 * Configure whether to ask for a guidance image
	 * @param askForGuidance
	 */
	public void setAskForGuidance(boolean askForGuidance) {
		this.askForGuidance = askForGuidance;
	}
	
	
	/**
	 * Gets the kernel width
	 * @return kernelWidth
	 */
	public int getKernelWidth() {
		return kernelWidth;
	}
	
	
	/**
	 * Computes a suggestion for the kernel's width
	 * exp(-0.5 * (distance(p1, p2) / sigmaGeom).^2 ) < kernelMinWeight, solved for distance(p1, p2)
	 * @return the suggested kernel width as an odd integer
	 */
	private int computeKernelWidth() {
		
		double maxSigmaGeom = Math.max(sigmaGeom[0], Math.max(sigmaGeom[1], sigmaGeom[2]));
		double width = Math.sqrt(-2 * Math.log(KERNEL_MIN_WEIGHT)) * maxSigmaGeom;
		
		// Kernel's width is usually an odd number
		int kernelWidth = (int) (width * 2);
		return (kernelWidth % 2 == 1) ? kernelWidth : kernelWidth + 1;
	}
	
	
	/**
	 * Pre-computes the geometric kernel which has size kernelWidth * kernelWidth * kernelWidth
	 * @return geometric kernel as Grid3D
	 */
	private Grid3D computeGeometricKernel() {
		
		assert(sigmaGeom.length == 3);
		assert(sigmaGeom[0] != 0);
		assert(sigmaGeom[1] != 0);
		assert(sigmaGeom[2] != 0);
		
		Grid3D geometricKernel = new Grid3D(kernelWidth, kernelWidth, kernelWidth);
		Point3d center = new Point3d(kernelWidth/2, kernelWidth/2, kernelWidth/2);
		
		for (int z=0; z<kernelWidth; z++) {
			for (int y=0; y<kernelWidth; y++) {
				for (int x=0; x<kernelWidth; x++) {
					Point3d current = new Point3d(x, y, z);
					double distance_x = (current.x - center.x) / sigmaGeom[0];
					double distance_y = (current.y - center.y) / sigmaGeom[1];
					double distance_z = (current.z - center.z) / sigmaGeom[2];
					
					float kernelVal = (float) Math.exp(-.5f * (distance_x * distance_x + distance_y * distance_y + distance_z * distance_z));
					
					geometricKernel.setAtIndex(x, y, z, kernelVal);
				}
			}
		}
		
		return geometricKernel;
	}
	
	
	/**
	 * Pre-computes the geometric kernel which has size kernelWidth * kernelWidth * kernelWidth
	 * Uses the kernel's symmetry to reduce runtime
	 * @return geometric kernel as Grid3D
	 */
	private Grid3D computeSymmetricGeometricKernel() {
		
		Grid3D geometricKernel = new Grid3D(kernelWidth, kernelWidth, kernelWidth);
		
		Point3d center = new Point3d(kernelWidth/2, kernelWidth/2, kernelWidth/2);
		
		for (int z=0; z<kernelWidth / 2 + 1; z++) {
			for(int y=0; y<kernelWidth / 2 + 1; y++) {
				for (int x=0; x<kernelWidth / 2 + 1; x++) {
					Point3d current = new Point3d(x, y, z);
					double distance = current.distanceTo(center);
					double distanceBySigmaGeom = distance / sigmaGeom[0];
					float kernelVal = (float) Math.exp(-.5f * distanceBySigmaGeom * distanceBySigmaGeom);
					
					// Math.max since upper bound of x (or y or z) is kernelWidth / 2 + 1
					geometricKernel.setAtIndex(x, y, z, kernelVal);
					geometricKernel.setAtIndex(Math.max((kernelWidth-1) - x, x), y, z, kernelVal);
					geometricKernel.setAtIndex(Math.max((kernelWidth-1) - x, x), Math.max((kernelWidth-1) - y, y), z, kernelVal);
					geometricKernel.setAtIndex(x, Math.max((kernelWidth-1) - y, y), z, kernelVal);
					
					geometricKernel.setAtIndex(x, y, Math.max((kernelWidth-1) - z, z), kernelVal);
					geometricKernel.setAtIndex(Math.max((kernelWidth-1) - x, x), y, Math.max((kernelWidth-1) - z, z), kernelVal);
					geometricKernel.setAtIndex(Math.max((kernelWidth-1) - x, x), Math.max((kernelWidth-1) - y, y), Math.max((kernelWidth-1) - z, z), kernelVal);
					geometricKernel.setAtIndex(x, Math.max((kernelWidth-1) - y, y), Math.max((kernelWidth-1) - z, z), kernelVal);					
				}
			}
		}
		
		return geometricKernel;
	}

	@Override
	protected void configureKernel() {
		filterKernel = program.createCLKernel(
				(showGuidance ? "jointBilateralFilter3D" : "bilateralFilter3D"));
		
		filterKernel
		.putArg(image)
		.putArg(result)
		.putArg(width)
		.putArg(height)
		.putArg(depth)
		.putArg(geometricKernel)
		.putArg(kernelWidth)
		.putArg((float)0.f)	// This value won't be used as we've precomputed and copied the geometric kernel onto the device
		.putArg((float)this.sigmaPhoto);
		
		if (showGuidance) {
			filterKernel
			.putArg(template)
			.putArg(1);	// Joint Bilateral Filter? Yes!
		}
	}
	

	
	@Override
	public void cleanup() {
		if (showGuidance && null != template) {
			template.release();
		}
		super.cleanup();
	}
	
	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Tomasi98-BFF,\n" +
		"  author = {Tomasi, C. and Manduchi, R.},\n" +
		"  title = {Bilateral Filtering for Gray and Color Images},\n" +
		"  booktitle = {ICCV '98: Proceedings of the Sixth International Conference on Computer Vision},\n" +
		"  year = {1998},\n" +
		"  isbn = {81-7319-221-9},\n" +
		"  pages = {839-846},\n" +
		"  publisher = {IEEE Computer Society},\n" +
		"  address = {Washington, DC, USA},\n" +
		"}\n";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Tomasi C, Maduchi R, Bilateral Filtering for Gray and Color Images. In: ICCV '98: Proceedings of the Sixth International Conference on Computer Vision, pp. 839-846, IEEE Computer Society, Washington, DC, United States 1998.";
	}


	@Override
	public boolean isDeviceDependent() {
		// Crashes on weak GPUs
		return true;
	}


	@Override
	public String getToolName() {
		return "OpenCL Bilateral Filter 3D";
	}


	@Override
	public void prepareForSerialization() {
		init = false;
	}

	@Override
	public ImageFilteringTool clone() {
		BilateralFiltering3DTool clone = new BilateralFiltering3DTool();
		clone.sigmaGeom = this.sigmaGeom;
		clone.sigmaPhoto = this.sigmaPhoto;
		clone.kernelWidth = this.kernelWidth;
		clone.showGuidance = this.showGuidance;
		clone.guidanceGrid = this.guidanceGrid;
		
		clone.configured = this.configured;
		return clone;
	}

}

/*
 * Copyright (C) 2018 Benedikt Lorch, Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
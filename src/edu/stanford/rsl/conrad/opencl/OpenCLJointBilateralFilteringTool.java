/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.opencl;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.MultiProjectionFilter;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class OpenCLJointBilateralFilteringTool extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1406305319277253826L;
	protected double sigmaPhoto = 1;
	protected double sigmaGeom = 5;
	protected boolean showGuidance = false;


	// initialized in init method.
	protected CLContext clContext;
	protected CLDevice device;
	protected CLKernel filterKernel;
	protected boolean init = false;
	protected CLBuffer<FloatBuffer> image;
	protected CLBuffer<FloatBuffer> template;
	protected CLBuffer<FloatBuffer> result;
	protected int width;
	protected int height;

	public synchronized void init(Grid2D grid){
		if (!init) {
			clContext = OpenCLUtil.createContext();
			device = clContext.getMaxFlopsDevice();
			OpenCLUtil.initFilter(clContext);

			init = true;
			width = grid.getWidth();
			height = grid.getHeight();

			// create memory
			template = clContext.createFloatBuffer(width*height, CLMemory.Mem.READ_ONLY);
			image = clContext.createFloatBuffer(width*height, CLMemory.Mem.READ_ONLY);
			result = clContext.createFloatBuffer(width*height, CLMemory.Mem.WRITE_ONLY);
			filterKernel = OpenCLUtil.filter.createCLKernel("bilateralFilter");

		}
	}

	public void release(){
		template.release();
		image.release();
		result.release();
		filterKernel.release();
		OpenCLUtil.releaseContext(clContext);
	}

	@Override
	public void configure() throws Exception {
		double sigmaGeom = UserUtil.queryDouble("Enter Geometric Sigma", this.sigmaGeom);
		double sigmaPhoto = UserUtil.queryDouble("Enter Photommetric Sigma", this.sigmaPhoto);
		boolean showGuidance = UserUtil.queryBoolean("Add guidance image in first channel?");
		configure(sigmaGeom, sigmaPhoto, showGuidance);
	}
	

	public void configure(double geom, double photo, boolean guide) throws Exception {
		context = 0;
		this.sigmaGeom = geom;
		this.sigmaPhoto = photo;
		this.showGuidance = guide;
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return "@inproceedings{Petschnigg04DPW, \n"+
		"  title={Digital photography with flash and no-flash image pairs},\n"+
		"  author={Petschnigg, Georg and Szeliski, Richard and Agrawala, Maneesh and Cohen, Michael and Hoppe, Hugues and Toyama, Kentaro},\n"+
		"  booktitle={ACM transactions on graphics (TOG)},\n"+
		"  volume={23},\n"+
		"  number={3},\n"+
		"  pages={664--672},\n"+
		"  year={2004},\n"+
		"  organization={ACM}\n"+
		"}";
	}

	@Override
	public String getMedlineCitation() {
		return "G. Petschnigg, M. Agrawala, H. Hoppe, R. Szeliski, M. Cohen, and K. Toyama, Digital photography with flash and no-flash image pairs, in Proc. ACM SIGGRAPH, 2004, pp. 664�672.";
	}

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		if (!init) init(inputQueue.get(projectionNumber));
		processOneProjection(projectionNumber);
	}
	
	protected void configureKernel(Grid2D grid, int channel){
		filterKernel
		.putArg(template)
		.putArg(image)
		.putArg(result)
		.putArg(grid.getWidth())
		.putArg(grid.getHeight())
		.putArg((float)this.sigmaGeom)
		.putArg((float)this.sigmaPhoto);
	}
	
	@Override
	protected void cleanup(){
		super.cleanup();
		release();
	}

	private synchronized void processOneProjection(int projectionNumber) throws Exception{
		Grid2D grid = inputQueue.get(projectionNumber);
		if (grid instanceof MultiChannelGrid2D){
			// JBF case
			MultiChannelGrid2D multi = (MultiChannelGrid2D)grid;

			// Create template
			Grid2D temp = new Grid2D(grid.getWidth(), grid.getHeight());
			for (int c = 0; c < multi.getNumberOfChannels(); c++){
				NumericPointwiseOperators.addBy(temp, multi.getChannel(c));
			}
			//float max=PointwiseOperators.max(temp);
			//float maxNoInf = PointwiseOperators.maxNoInf(temp);

			CLCommandQueue queue = device.createCommandQueue();
			template.getBuffer().put(temp.getBuffer());
			template.getBuffer().rewind();
			queue.putWriteBuffer(template, false);

			// Apply Filter
			for (int c=0; c <multi.getNumberOfChannels(); c++){
				image.getBuffer().put(multi.getChannel(c).getBuffer());
				image.getBuffer().rewind();
				queue.putWriteBuffer(image, true);	
				filterKernel.rewind();
				configureKernel(multi.getChannel(c), c);
				int groupSize = Math.min(device.getMaxWorkGroupSize(), 16);
				queue.put2DRangeKernel(filterKernel, 0, 0, OpenCLUtil.roundUp(groupSize, grid.getWidth()), OpenCLUtil.roundUp(groupSize, grid.getHeight()), groupSize, groupSize);
				queue.flush();
				queue.finish();
				queue.putReadBuffer(result, true);
				result.getBuffer().get(multi.getChannel(c).getBuffer());
				result.getBuffer().rewind();	
			}
			queue.release();

			if (showGuidance){
				MultiChannelGrid2D out = new MultiChannelGrid2D(multi.getWidth(), multi.getHeight(), multi.getNumberOfChannels()+1);
				String [] channelNames = new String[multi.getChannelNames().length+1];
				for (int c=0;c<multi.getNumberOfChannels();c++){
					out.setChannel(c+1, multi.getChannel(c));
					channelNames[c+1] = multi.getChannelNames()[c];
				}
				channelNames[0] = "Guidance Image";
				out.setChannel(0, temp);
				out.setChannelNames(channelNames);
				grid = out;
			}

		} else {
			// BF Case
			CLCommandQueue queue = device.createCommandQueue();
			template.getBuffer().put(grid.getBuffer());
			template.getBuffer().rewind();
			queue.putWriteBuffer(template, false);
			image.getBuffer().put(grid.getBuffer());
			image.getBuffer().rewind();
			queue.putWriteBuffer(image, true);	
			filterKernel.rewind();
			configureKernel(grid, 0);
			int groupSize = Math.min(device.getMaxWorkGroupSize(), 16);
			queue.put2DRangeKernel(filterKernel, 0, 0, OpenCLUtil.roundUp(groupSize, grid.getWidth()), OpenCLUtil.roundUp(groupSize, grid.getHeight()), groupSize, groupSize);
			queue.flush();
			queue.finish();
			queue.putReadBuffer(result, true);
			result.getBuffer().get(grid.getBuffer());
			result.getBuffer().rewind();
			queue.release();
		}
		sink.process(grid, projectionNumber);
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		return "OpenCL Joint Bilateral Filter";
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	/**
	 * @return the sigmaPhoto
	 */
	public double getSigmaPhoto() {
		return sigmaPhoto;
	}

	/**
	 * @param sigmaPhoto the sigmaPhoto to set
	 */
	public void setSigmaPhoto(double sigmaPhoto) {
		this.sigmaPhoto = sigmaPhoto;
	}

	/**
	 * @return the sigmaGeom
	 */
	public double getSigmaGeom() {
		return sigmaGeom;
	}

	/**
	 * @param sigmaGeom the sigmaGeom to set
	 */
	public void setSigmaGeom(double sigmaGeom) {
		this.sigmaGeom = sigmaGeom;
	}


	/**
	 * @return the showGuidance
	 */
	public boolean isShowGuidanceImage() {
		return showGuidance;
	}

	/**
	 * @param showGuidance the showGuidance to set
	 */
	public void setShowGuidanceImage(boolean showGuidance) {
		this.showGuidance = showGuidance;
	}

}

package edu.stanford.rsl.conrad.filtering.opencl;

import com.jogamp.opencl.CLCommandQueue;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.opencl.OpenCLFilteringTool3D;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Tool for OpenCL computation of the 2D Mean Filter in each slice of a Grid 3D.
 * This tool uses the CONRAD internal Grid 3-D and Grid 2-D data structure.
 *   
 * @author Jennifer Maier
 *
 */

public class MeanFiltering2DinGrid3DTool extends OpenCLFilteringTool3D {

	private static final long serialVersionUID = -1469783337390366468L;
	protected int kernelWidth = 3;
	protected int kernelHeight = 3;

	public MeanFiltering2DinGrid3DTool() {
		this.kernelName = kernelname.MEAN_FILTER_2D_IN_GRID3D;
	}

	@Override
	public void configure() throws Exception {		

		kernelWidth = UserUtil.queryInt("Enter kernel width\n(even numbers will be incremented by 1)", kernelWidth);
		if (kernelWidth % 2 == 0) {
			kernelWidth++;
		}

		kernelHeight = UserUtil.queryInt("Enter kernel width\n(even numbers will be incremented by 1)", kernelHeight);
		if (kernelHeight % 2 == 0) {
			kernelHeight++;
		}

		configured = true;

	}
	
	// Getter and Setter
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}

	public int getKernelWidth() {
		return kernelWidth;
	}

	public void setKernelWidth(int kernelWidth) {
		if (kernelWidth % 2 == 0) {
			kernelWidth++;
		}
		this.kernelWidth = kernelWidth;
	}

	public int getKernelHeight() {
		return kernelHeight;
	}

	public void setKernelHeight(int kernelHeight) {
		if (kernelHeight % 2 == 0) {
			kernelHeight++;
		}
		this.kernelHeight = kernelHeight;
	}		

	@Override
	protected void configureKernel() {
		filterKernel = program.createCLKernel("meanFilter2DinGrid3D");

		filterKernel
		.putArg(image)
		.putArg(result)
		.putArg(width)
		.putArg(height)
		.putArg(depth)
		.putArg(kernelWidth)
		.putArg(kernelHeight)
		.putArg(1.0f/(kernelWidth*kernelHeight));			

	}
	
	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getToolName() {
		return "OpenCL Mean Filter 2D slices in Grid3D";
	}

	@Override
	public ImageFilteringTool clone() {
		MeanFiltering2DinGrid3DTool clone = new MeanFiltering2DinGrid3DTool();
		clone.setKernelHeight(this.getKernelHeight());
		clone.setKernelWidth(this.getKernelWidth());
		clone.setConfigured(this.configured);

		return clone;
	}

}

package edu.stanford.rsl.conrad.filtering.opencl;

import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Tool for OpenCL 2D Mean Filtering a Grid 2D.
 * This tool uses the CONRAD internal Grid 2-D data structure.
 *   
 * @author Jennifer Maier
 *
 */
public class MeanFiltering2DTool extends OpenCLFilteringTool2D {

	private static final long serialVersionUID = -3381321981652625848L;
	protected int kernelWidth = 3;
	protected int kernelHeight = 3;
	
	public MeanFiltering2DTool() {
		this.kernelName = kernelname.MEAN_FILTER_2D;
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
		filterKernel = program.createCLKernel("meanFilter2D");

		filterKernel
		.putArg(image)
		.putArg(result)
		.putArg(width)
		.putArg(height)
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
		return "OpenCL Mean Filter 2D";
	}

	@Override
	public ImageFilteringTool clone() {
		MeanFiltering2DTool clone = new MeanFiltering2DTool();
		clone.setKernelHeight(this.getKernelHeight());
		clone.setKernelWidth(this.getKernelWidth());
		clone.setConfigured(this.configured);

		return clone;
	}
	
}

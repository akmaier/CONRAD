package edu.stanford.rsl.conrad.filtering.multiprojection.blocks;

import edu.stanford.rsl.conrad.filtering.opencl.BilateralFiltering3DTool;
import edu.stanford.rsl.conrad.filtering.opencl.OpenCLFilteringTool3D;

/**
 * Image processing block for applying the Bilateral Filter to larger images such that the image to be filtered needs to be split up in blocks
 * @author Benedikt Lorch
 *
 */
public class BilateralFilter3DCLBlock extends ImageProcessingBlock {

	private static final long serialVersionUID = 7627475721714651395L;	
	private BilateralFiltering3DTool bilateralFilteringTool;

	
	public OpenCLFilteringTool3D getFilteringTool() {
		return bilateralFilteringTool;
	}
	
	
	public BilateralFilter3DCLBlock() {
		this.bilateralFilteringTool = new BilateralFiltering3DTool();
	}
	
	@Override
	public ImageProcessingBlock clone() {
		BilateralFilter3DCLBlock clone = new BilateralFilter3DCLBlock();
		clone.bilateralFilteringTool = (BilateralFiltering3DTool) this.bilateralFilteringTool.clone();
		return clone;
	}

	@Override
	protected void processImageBlock() {
		int[] size = inputBlock.getSize();
		
		bilateralFilteringTool.init(size[0], size[1], size[2]);
		outputBlock = bilateralFilteringTool.process(inputBlock);
		bilateralFilteringTool.cleanup();
	}


	public void configure() throws Exception {
		bilateralFilteringTool.configure();
		configured = true;
	}

}

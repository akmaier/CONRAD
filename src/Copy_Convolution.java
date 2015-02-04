import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;


/**
 * Filters the projection data with a 2-D filter which is selected via the ImageJ GUI.
 * 
 * @author Andreas Maier
 *
 */
public class Copy_Convolution implements PlugIn {

	public Copy_Convolution(){
		
	}
	
	@Override
	public void run(String arg) {
		Configuration.loadConfiguration();
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		ImagePlus filtered = (ImagePlus) JOptionPane.showInputDialog(null, "Select image to mimic: ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		ImagePlus notFiltered = (ImagePlus) JOptionPane.showInputDialog(null, "Select image to adjust: ", "Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		try {
			FloatProcessor before = (FloatProcessor) filtered.getChannelProcessor().duplicate();
			FloatProcessor after = (FloatProcessor) notFiltered.getChannelProcessor().duplicate();
			ImageUtil.normalizeImageProcessorMinMax(before);
			ImageUtil.normalizeImageProcessorMinMax(after);
			int kernelSize = 5;
			float []  kernel = ImageUtil.estimateConvolutionKernel(before, after, kernelSize, 10000);
			for (int i=0;i<notFiltered.getStackSize();i++){
				notFiltered.getStack().getProcessor(i+1).convolve(kernel, kernelSize, kernelSize);
				IJ.showStatus("Applying Convolution");
				IJ.showProgress((i + 0.0) / notFiltered.getStackSize());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			IJ.log(e.getLocalizedMessage());
		}
	
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

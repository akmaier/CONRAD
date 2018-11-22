package edu.stanford.rsl.conrad.angio.reconstruction.autofocus;

import ij.ImagePlus;
import ij.process.StackStatistics;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class HistoAutofocus {

	private static double acceptanceThreshold = 1.0f;
	private static double gain = 10.0f;
	
	public static float evaluateAutoFocus(Grid3D img, float threshold){
		ImagePlus imp = ImageUtil.wrapGrid3D(img, "");
		StackStatistics stats = new StackStatistics(imp, 256, 0, threshold);
		long[] hist = stats.getHistogram();
		float val = 0;
		for(int i = 0; i < hist.length; i++){
			double binVal = (i+1f)*stats.binSize;
			float denom = (float)((binVal<acceptanceThreshold)?binVal/gain:Math.pow(binVal,2));
			val += hist[i] / denom;
		}
		val /= hist.length;
		return val;	
	}
	
}

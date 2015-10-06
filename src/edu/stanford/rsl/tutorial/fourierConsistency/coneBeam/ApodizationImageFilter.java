package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.numerics.DoubleFunction;
import edu.stanford.rsl.conrad.utils.Configuration;

public class ApodizationImageFilter extends IndividualImageFilteringTool {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4248650827151829457L;

	private DoubleFunction[] apodFctsU;
	private DoubleFunction[] apodFctsV;
	private boolean initialized = false;

	private windowType wtU;
	private windowType wtV;

	private Integer[] customSizes = null;

	public Integer[] getCustomSizes() {
		return customSizes;
	}

	public void setCustomSizes(Integer[] customSizes) {
		this.customSizes = customSizes;
	}

	private Configuration config;

	public ApodizationImageFilter() {
	}

	public ApodizationImageFilter(ApodizationImageFilter in) {
		this.wtU = in.wtU;
		this.wtV = in.wtV;
		this.config = in.config;
		this.initialized = false;
		this.customSizes = in.customSizes;
	}

	public enum windowType{
		rect(0),
		hann(1),
		hamming(2),
		cosine(3),
		blackman(4),
		blackmanHarris(5),
		triangular(6),
		welch(7);

		private int label;
		private windowType(int lab){
			this.label = lab;
		}

		public int getLabel(){
			return this.label;
		}
	}

	private void initFcts(){
		int U = config.getGeometry().getDetectorWidth();
		int V = config.getGeometry().getDetectorHeight();

		int[] s = new int[]{U,V};
		for (int i = 0; i < s.length; i++) {
			int sin = s[i];
			int N = (customSizes != null && customSizes[i] != null) ? customSizes[i] : sin;
			int offset = (customSizes != null && customSizes[i] != null) ? (sin-customSizes[i])/2 : 0;
			int blockSize = (customSizes != null && customSizes[i] != null) ? customSizes[i] : s[i];
			DoubleFunction fn = (customSizes != null && customSizes[i] != null) ? nin -> (nin < offset) ? 0 : ( (nin >= (offset + blockSize) ) ? (blockSize -1) :  (nin - offset) ) : nin -> nin;

			DoubleFunction[] fcts = new DoubleFunction[windowType.values().length];
			fcts[windowType.rect.getLabel()] = n -> Math.max(1, 0); // rect
			fcts[windowType.triangular.getLabel()] = n -> Math.max(1-Math.abs((fn.f(n)-((N-1)/2))/(N/2)),0); // triangular
			fcts[windowType.welch.getLabel()] = n -> Math.max(1-Math.pow((fn.f(n)-((N-1)/2))/((N-1)/2),2),0); // welch
			fcts[windowType.hann.getLabel()] = n -> Math.max(0.5*(1-Math.cos(2*Math.PI*fn.f(n)/(N-1))),0); // hann
			fcts[windowType.hamming.getLabel()] = n -> Math.max(0.54 - 0.46 * Math.cos(2*Math.PI*fn.f(n)/(N-1)),0); // hamming
			fcts[windowType.blackman.getLabel()] = n -> Math.max(0.42 - 0.50 * Math.cos(2*Math.PI*fn.f(n)/(N-1)) + 0.08 * Math.cos(4*Math.PI*fn.f(n)/(N-1)),0); // blackman
			fcts[windowType.blackmanHarris.getLabel()] = n -> Math.max(0.35875 - 0.48829 * Math.cos(2*Math.PI*fn.f(n)/(N-1)) + 0.14128 * Math.cos(4*Math.PI*fn.f(n)/(N-1)) - 0.01168 * Math.cos(6*Math.PI*fn.f(n)/(N-1)) ,0); // blackman-harris
			fcts[windowType.cosine.getLabel()] = n -> Math.max(Math.sin(Math.PI*fn.f(n)/(N-1)),0); // cosine

			if(i==0)
				apodFctsU = fcts;
			else
				apodFctsV = fcts;
		}
	}

	@Override
	public void configure() throws Exception {
		config = Configuration.getGlobalConfiguration();
		wtU = windowType.rect;
		wtV = windowType.blackman;
	}

	@Override
	public String getBibtexCitation() {
		return null;
	}

	@Override
	public String getMedlineCitation() {
		return null;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		return new ApodizationImageFilter(this);
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		if(!initialized)
			initFcts();
		for (int i = 0; i < imageProcessor.getSize()[0]; i++) {
			for (int j = 0; j < imageProcessor.getSize()[1]; j++) {
				imageProcessor.setAtIndex(i, j, 
						(float)(imageProcessor.getAtIndex(i, j)*
								apodFctsU[wtU.getLabel()].f(i)*
								apodFctsV[wtV.getLabel()].f(j)));
			}
		}
		return imageProcessor;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		return "Apodization Window Filter";
	}

}

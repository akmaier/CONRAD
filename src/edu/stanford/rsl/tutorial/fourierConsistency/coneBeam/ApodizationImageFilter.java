/*
 * Copyright (C) 2015 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
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
	private DoubleFunction[] apodFctsK;
	private boolean initialized = false;

	private windowType wtU;

	private windowType wtV;

	private windowType wtK;

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
		this.wtK = in.wtK;
		this.config = in.config;
		this.initialized = false;
		this.customSizes = in.customSizes;
	}

	public static enum windowType{
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
		int K = config.getGeometry().getNumProjectionMatrices();

		int[] s = new int[]{U,V,K};
		for (int i = 0; i < s.length; i++) {
			int sin = s[i];
			final int N = (customSizes != null && customSizes[i] != null) ? customSizes[i] : sin;
			final int offset = (customSizes != null && customSizes[i] != null) ? (sin-customSizes[i])/2 : 0;
			final int blockSize = (customSizes != null && customSizes[i] != null) ? customSizes[i] : s[i];
			final int j = i;

			final DoubleFunction fnRect = new DoubleFunction() {
				@Override
				public double f(double x) {
					if(customSizes != null && customSizes[j] != null){
						if(x < offset){
							return Double.NaN;
						}
						else if(x >= (offset + blockSize)){
							return (Double.NaN);
						}
						else{
							return x - offset;
						}						
					}
					else{
						return x;
					}
				}
			};

			final DoubleFunction fn = new DoubleFunction() {
				@Override
				public double f(double x) {
					if(customSizes != null && customSizes[j] != null){
						if(x < offset){
							return 0;
						}
						else if(x >= (offset + blockSize)){
							return (blockSize - 1);
						}
						else{
							return x - offset;
						}						
					}
					else{
						return x;
					}
				}
			};

			// writing out lambda expressions to keep Java 1.7 support
			DoubleFunction[] fcts = new DoubleFunction[windowType.values().length];
			fcts[windowType.rect.getLabel()] = new DoubleFunction() {
				public double f(double x) {
					if (Double.isNaN(fnRect.f(x)))
						return 0.0;
					else
						return 1.0;
				}
			};
			fcts[windowType.triangular.getLabel()] =  new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(1-Math.abs((fn.f(x)-((N-1)/2))/(N/2)),0); // triangular
				}
			};
			fcts[windowType.welch.getLabel()] = new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(1-Math.pow((fn.f(x)-((N-1)/2))/((N-1)/2),2),0); // welch
				}
			};
			fcts[windowType.hann.getLabel()] = new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(0.5*(1-Math.cos(2*Math.PI*fn.f(x)/(N-1))),0); // hann
				}
			};
			fcts[windowType.hamming.getLabel()] = new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(0.54 - 0.46 * Math.cos(2*Math.PI*fn.f(x)/(N-1)),0); // hamming
				}
			};
			fcts[windowType.blackman.getLabel()] = new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(0.42 - 0.50 * Math.cos(2*Math.PI*fn.f(x)/(N-1)) + 0.08 * Math.cos(4*Math.PI*fn.f(x)/(N-1)),0); // blackman
				}
			};
			fcts[windowType.blackmanHarris.getLabel()] = new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(0.35875 - 0.48829 * Math.cos(2*Math.PI*fn.f(x)/(N-1)) + 0.14128 * Math.cos(4*Math.PI*fn.f(x)/(N-1)) - 0.01168 * Math.cos(6*Math.PI*fn.f(x)/(N-1)) ,0); // blackman-harris
				}
			};
			fcts[windowType.cosine.getLabel()] = new DoubleFunction() {
				@Override
				public double f(double x) {
					return Math.max(Math.sin(Math.PI*fn.f(x)/(N-1)),0); // cosine
				}
			}; 

			switch (i) {
			case 0:
				apodFctsU = fcts;
				break;
			case 1:
				apodFctsV = fcts;
				break;
			case 2:
				apodFctsK = fcts;
				break;

			default:
				apodFctsU = fcts;
				break;
			}
		}
	}

	@Override
	public void configure() throws Exception {
		config = Configuration.getGlobalConfiguration();
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
								apodFctsV[wtV.getLabel()].f(j)*
								apodFctsK[wtK.getLabel()].f(this.getImageIndex())
								));
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


	public windowType getWtU() {
		return wtU;
	}

	public void setWtU(windowType wtU) {
		this.wtU = wtU;
		initialized = false;
	}

	public windowType getWtV() {
		return wtV;
	}

	public void setWtV(windowType wtV) {
		this.wtV = wtV;
		initialized = false;
	}
	
	public windowType getWtK() {
		return wtK;
	}

	public void setWtK(windowType wtK) {
		this.wtK = wtK;
		initialized = false;
	}


}

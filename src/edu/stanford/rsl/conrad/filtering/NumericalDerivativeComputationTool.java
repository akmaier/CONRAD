package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.process.FloatProcessor;

/**
 * Tool to compute first derivate numerically.
 * 
 * @author Andreas Maier
 *
 */
public class NumericalDerivativeComputationTool extends IndividualImageFilteringTool {


	/**
	 * 
	 */
	private static final long serialVersionUID = 1549628690729735281L;
	private int nTimes = 3;
	private boolean twoD = false;
	private boolean horizontal = true;

	public NumericalDerivativeComputationTool (){
		configured = true;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		NumericalDerivativeComputationTool filter = new NumericalDerivativeComputationTool();
		filter.configured = configured;
		filter.nTimes = nTimes;
		filter.twoD = twoD;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Numerical Derivative (contex = " + nTimes + ")";
	}

	private float [] designKernel(int width){
		if (width < 2) width = 2;
		if (width == 2) {return new float[] {-1, 1, 0};}
		if (width % 2 == 0) width ++;
		float [] revan = new float[width];
		for (int i = 1; i<=width/2;i++){
			revan[((width)/2) + i] = i;
			revan[((width)/2) - i] = -i;
		}	
		return revan;
	}
	
	private float [] designKernelX(){
		return new float [] {1, 2, 1, 0, 0, 0, -1, -2 ,-1};
	}
	
	private float [] designKernelY(){
		return new float [] {1, 0, -1, 2, 0, -2, 1, 0 ,-1};
	}
	

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		FloatProcessor revan = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		revan.setPixels(imageProcessor.getBuffer());
		if (!twoD){
			float [] kernel = designKernel(nTimes);
			if (horizontal)
				revan.convolve(kernel, kernel.length, 1);
			else 
				revan.convolve(kernel, 1, kernel.length);
			revan.multiply(0.5/Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX());
		} else {
			FloatProcessor xDerivative = (FloatProcessor) revan.duplicate();
			xDerivative.convolve(designKernelX(), nTimes, nTimes);
			xDerivative.multiply(0.5/Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX());
			xDerivative.sqr();
			FloatProcessor yDerivative = (FloatProcessor) revan.duplicate();
			yDerivative.convolve(designKernelY(), nTimes, nTimes);
			yDerivative.sqr();
			yDerivative.multiply(0.5/Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionY());
			ImageUtil.addProcessors(xDerivative, yDerivative);
			xDerivative.sqrt();
			revan = xDerivative;
		}
		Grid2D result = new Grid2D((float[]) revan.getPixels(), revan.getWidth(), revan.getHeight());
		return result;
	}

	@Override
	public void configure() throws Exception {
		nTimes = UserUtil.queryInt("Enter context for derivative computation: ", nTimes);
		twoD = UserUtil.queryBoolean("Compute in both directions?");
		if (!twoD) {
			horizontal = UserUtil.queryBoolean("Compute horizontal?");
		}
		if (twoD) nTimes = 3;
		configured = true;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	/**
	 * @return the nTimes
	 */
	public int getnTimes() {
		return nTimes;
	}

	/**
	 * @param nTimes the nTimes to set
	 */
	public void setnTimes(int nTimes) {
		this.nTimes = nTimes;
	}

	/**
	 * @return the twoD
	 */
	public boolean isTwoD() {
		return twoD;
	}

	/**
	 * @param twoD the twoD to set
	 */
	public void setTwoD(boolean twoD) {
		this.twoD = twoD;
	}

	/**
	 * @return the horizontal
	 */
	public boolean isHorizontal() {
		return horizontal;
	}

	/**
	 * @param horizontal the horizontal to set
	 */
	public void setHorizontal(boolean horizontal) {
		this.horizontal = horizontal;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

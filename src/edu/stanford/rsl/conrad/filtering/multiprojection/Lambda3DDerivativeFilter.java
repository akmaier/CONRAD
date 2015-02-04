package edu.stanford.rsl.conrad.filtering.multiprojection;


import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Class for the computation of a derivative along the source trajectory. (Used in super-short-scan reconstruction algorithms.)
 * @author akmaier
 *
 */
public class Lambda3DDerivativeFilter extends MultiProjectionFilter {


	/**
	 * 
	 */
	private static final long serialVersionUID = -4713499234279226464L;
	/**
	 * 
	 */
	
	double [] uWeights = null;
	double [][] vWeights = null;
	double [] pWeights = null;
	double detectorElementSizeX = 1;
	double detectorElementSizeY = 1;
	double averageAngularIncrement = 1;
	double factor =  1;
	double factor2 = 1;
	double SID = 1;
	
	/**
	 * Creates a new lambda derivative, i.e. derivative is computed along the source trajectory lambda.
	 */
	public Lambda3DDerivativeFilter(){
		context = 1;
		configured = false;
	}

	@Override
	protected synchronized void processProjectionData(int projectionNumber) throws Exception {
		//if (uWeights == null || projectionNumber == 0) 
		init(projectionNumber);
		//if (debug > 0) System.out.println("ProjectionSortingFilter: Projecting " + projectionNumber);
		Grid2D derivative = computeUVDerivative(projectionNumber);
		if (projectionNumber > 0) { // cannot compute the lambda derivative in the first projection
			// compute lambda derivative
			//ImageProcessor lambdaDerivative = inputQueue.get(projectionNumber-1).duplicate();
			//ImageUtil.subtractProcessors(lambdaDerivative, inputQueue.get(projectionNumber));
			Grid2D lambdaDerivative = new Grid2D(inputQueue.get(projectionNumber-1));
			Grid2D result = new Grid2D(lambdaDerivative.getWidth(), lambdaDerivative.getHeight());
			result = (Grid2D)NumericPointwiseOperators.subtractedBy(lambdaDerivative, inputQueue.get(projectionNumber));
			//ImageProcessor two = inputQueue.get(projectionNumber);
			float [] pixels = (float[]) result.getPixels();
			float [] pixels2 = (float[]) derivative.getPixels();
			double correction = (SID *(Math.tan(averageAngularIncrement * Math.PI / 180.0)));
			//System.out.println(detectorElementSizeX + " " + averageAngularIncrement + " " +correction);
			for (int j=0; j < lambdaDerivative.getHeight(); j++){
				for(int i =0; i < lambdaDerivative.getWidth(); i++){			
					//pixels [i+(j*lambdaDerivative.getWidth())] -= two.getInterpolatedValue(i, j);
					pixels [i+(j*lambdaDerivative.getWidth())] /= correction;
					pixels [i+(j*lambdaDerivative.getWidth())] *= factor;
					pixels2 [i+(j*lambdaDerivative.getWidth())] *= factor2;
					pixels2 [i+(j*lambdaDerivative.getWidth())] += pixels [i+(j*lambdaDerivative.getWidth())];
				}
			}
		}
		float [] pixels = (float[]) derivative.getPixels();
		for (int j=0; j < derivative.getHeight(); j++){
			for(int i =0; i < derivative.getWidth(); i++){
				pixels [i+(j*derivative.getWidth())] *= pWeights[i];
			}
		}
		// write to output queue
		sink.process(derivative, projectionNumber);
		if (projectionNumber > 0) inputQueue.remove(projectionNumber-1);
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	@Override
	public String getToolName() {
		return "3D Lambda Derivative Filter";
	}

	private Grid2D computeUVDerivative(int index){
		float [] kernel = {-1, 0, 1};
		ImageProcessor revan = new FloatProcessor(inputQueue.get(index).getWidth(), inputQueue.get(index).getHeight());
		// derivative in row direction
		FloatProcessor uDev = new FloatProcessor(revan.getWidth(), revan.getHeight());
		System.arraycopy(inputQueue.get(index).getBuffer(), 0, (float[]) uDev.getPixels(), 0, inputQueue.get(index).getBuffer().length);
		uDev.convolve(kernel, 3, 1);
		// derivative in column direction
		ImageProcessor vDev = new FloatProcessor(revan.getWidth(), revan.getHeight());
		System.arraycopy(inputQueue.get(index).getBuffer(), 0, (float[]) vDev.getPixels(), 0, inputQueue.get(index).getBuffer().length);
		vDev.convolve(kernel, 1, 3);
		// weight and add to result.
		for(int i =0; i<uDev.getWidth(); i++){
			for (int j=0; j< uDev.getHeight(); j++){
				double u = uDev.getPixelValue(i, j);
				u /= detectorElementSizeX * 2;
				u *= uWeights[i];
				uDev.putPixelValue(i, j, u);
				double v = vDev.getPixelValue(i, j);
				v *= vWeights[i][j];
				v /= detectorElementSizeY * 2;
				revan.putPixelValue(i, j, v+u);
			}
		}
		//if (index == 1) VisualizationUtil.showImageProcessor(revan);
		return new Grid2D((float[])revan.getPixels(), revan.getWidth(), revan.getHeight());
	}
	
	@Override
	public boolean isDeviceDependent() {
		// surely not.
		return false;
	}

	private synchronized void init(int projectionNumber){
		if (inputQueue.get(projectionNumber) == null) {
			throw new RuntimeException("Access violation " + projectionNumber);
		}
		
		double offsetU = inputQueue.get(projectionNumber).getWidth()/2;
		double offsetV = inputQueue.get(projectionNumber).getHeight()/2;
		double SID = this.SID;
		try {
			// compute parameters using projection geometry
			Trajectory geometry = Configuration.getGlobalConfiguration().getGeometry();
			offsetU = geometry.getProjectionMatrix(projectionNumber).getK().getElement(0, 3);
			offsetV = geometry.getProjectionMatrix(projectionNumber).getK().getElement(1, 3);
			SimpleVector spacingUV = new SimpleVector(geometry.getPixelDimensionX(), geometry.getPixelDimensionY());
			SID = geometry.getProjectionMatrix(projectionNumber).computeSourceToDetectorDistance(spacingUV)[0]; 
		} catch (Exception e) {
			// No geometry for these projections...
		}
		uWeights = new double[inputQueue.get(projectionNumber).getWidth()];
		for (int i=0; i<uWeights.length;i++){
			double pos = (i-(offsetU)) * detectorElementSizeX;
			uWeights[i]= ((Math.pow(pos,2) + Math.pow(SID, 2)) / SID);
		}
		vWeights = new double[inputQueue.get(projectionNumber).getWidth()][inputQueue.get(projectionNumber).getHeight()];
		for (int j=0; j<vWeights[0].length;j++){
			for (int i=0; i<vWeights.length;i++){
				double posU = (i-(offsetU)) * detectorElementSizeX;
				double posV = (j-(offsetV)) * detectorElementSizeY;
				vWeights[i][j]= (posU * posV) / SID;
				//vWeights[i][j] = 0;
			}
		}
		pWeights  = new double [inputQueue.get(projectionNumber).getWidth()];
		for (int i=0; i<pWeights.length;i++){
			double pos = (i-(offsetU)) * detectorElementSizeX;
			pWeights[i]= SID / Math.sqrt(Math.pow(pos,2) + Math.pow(SID, 2));
		}
	}
	
	@Override
	public void configure() throws Exception {
		Configuration config = Configuration.getGlobalConfiguration();
		SID = config.getGeometry().getSourceToDetectorDistance();
		detectorElementSizeX = config.getGeometry().getPixelDimensionX();
		detectorElementSizeY = config.getGeometry().getPixelDimensionY();
		averageAngularIncrement = config.getGeometry().getAverageAngularIncrement();
		pWeights = null;
		uWeights = null;
		vWeights = null;
		//factor = UserUtil.queryDouble("factor", factor);
		//factor2 = UserUtil.queryDouble("factor2", factor2);
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return "see Medline.";
	}

	@Override
	public String getMedlineCitation() {
		return "Yu H, Wang G. Feldkamp-type VOI reconstruction from super-short-scan cone-beam data. Med Phys 31(6):1357-62. 2004.";
	}
	
	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		uWeights = null;
		vWeights = null;
		pWeights = null;
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

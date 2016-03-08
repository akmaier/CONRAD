package edu.stanford.rsl.conrad.filtering.redundancy;


import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;



/**
 * Implementation of Parker weights. Note that you may require a HorizontalFlippingTool applied before and after the weights depending on the rotation direction of the source detector pair.
 * @author Andreas Maier
 *
 */
public class ParkerWeightingTool extends IndividualImageFilteringTool {

	private static final long serialVersionUID = -1342759492107420854L;
	
	protected int detectorWidth = 0;
	protected double pixelDimensionX = 0;
	protected double sourceToDetectorDistance = 0;
	protected double sourceToAxisDistance = 0;
	protected int numberOfProjections = 0;
	private double [] primaryAngles = null;
	protected double delta = 0;
	protected double offset = 0;
	protected static boolean debug = false;

	public ParkerWeightingTool()
	{
		super();
	}
	
	public ParkerWeightingTool(Trajectory g)
	{
		super();
		sourceToDetectorDistance = g.getSourceToDetectorDistance();
		pixelDimensionX = g.getPixelDimensionX();
		detectorWidth = g.getDetectorWidth();
		primaryAngles = normalizePrimaryAngleRange(g.getPrimaryAngles());
		numberOfProjections = g.getNumProjectionMatrices();
	}	
	
	public int getNumberOfProjections() {
		return numberOfProjections;
	}

	public void setNumberOfProjections(int numberOfProjections) {
		this.numberOfProjections = numberOfProjections;
	}

	public int getDetectorWidth() {
		return detectorWidth;
	}

	public void setDetectorWidth(int detectorWidth) {
		this.detectorWidth = detectorWidth;
	}

	public double getPixelDimensionX() {
		return pixelDimensionX;
	}

	public void setPixelDimensionX(double pixelDimensionX) {
		this.pixelDimensionX = pixelDimensionX;
	}

	public double getSourceToDetectorDistance() {
		return sourceToDetectorDistance;
	}

	public void setSourceToDetectorDistance(double sourceToDetectorDistance) {
		this.sourceToDetectorDistance = sourceToDetectorDistance;
	}

	public double[] getPrimaryAngles() {
		return primaryAngles;
	}

	public void setPrimaryAngles(double[] primaryAngles) {
		this.primaryAngles = normalizePrimaryAngleRange(primaryAngles);
	}
	
	public double[] normalizePrimaryAngleRange(double[] angles){
		double[] newArray = new double[angles.length];
		System.arraycopy(angles, 0, newArray, 0, newArray.length);
		
		// normalize to [0,360[ range --> all angles are in this range afterwards
		for (int i = 0; i < newArray.length; i++) {
			newArray[i] = newArray[i]%360;
			if (newArray[i] < 0)
				newArray[i]+=360;
		}
		
		// rotate all angles such that the array maximum is minimal
		int rowIdx = 0;
		double min = Double.POSITIVE_INFINITY;
		SimpleMatrix tmp = new SimpleMatrix(newArray.length,newArray.length);
		for (int i = 0; i < newArray.length; i++) {
			
			// rotate angles such that the i-th value has angle 0 degrees
			for (int j = 0; j < newArray.length; j++) {
					tmp.setElementValue(i, j, (newArray[j]-newArray[i])%360);
					if (tmp.getElement(i, j) < 0) 
						tmp.setElementValue(i, j, tmp.getElement(i, j)+360);
			}
			// find the minimum of the maximum value of all rotations
			if (tmp.getRow(i).max() < min){
				min = tmp.getRow(i).max();
				rowIdx = i;
			}
		}
		
		// return the best 
		return tmp.getRow(rowIdx).copyAsDoubleArray();
	}

	public void setConfiguration(Configuration config){
		sourceToDetectorDistance = config.getGeometry().getSourceToDetectorDistance();
		pixelDimensionX = config.getGeometry().getPixelDimensionX();
		detectorWidth = config.getGeometry().getDetectorWidth();
		primaryAngles = normalizePrimaryAngleRange(config.getGeometry().getPrimaryAngles());
		numberOfProjections = config.getGeometry().getNumProjectionMatrices();
	}

	public void setConfiguration(Trajectory g){
		sourceToDetectorDistance = g.getSourceToDetectorDistance();
		pixelDimensionX = g.getPixelDimensionX();
		detectorWidth = g.getDetectorWidth();
		primaryAngles = normalizePrimaryAngleRange(g.getPrimaryAngles());
		numberOfProjections = g.getNumProjectionMatrices();
	}	
	
	@Override
	public IndividualImageFilteringTool clone() {
		ParkerWeightingTool clone = new ParkerWeightingTool();
		clone.setDetectorWidth(this.getDetectorWidth());
		clone.setPixelDimensionX(this.getPixelDimensionX());
		clone.setSourceToDetectorDistance(this.getSourceToDetectorDistance());
		clone.setNumberOfProjections(numberOfProjections);
		clone.setPrimaryAngles(primaryAngles);
		clone.configured = configured;
		clone.offset = offset;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Parker Redundancy Weighting Filter (uses primary angles)";
	}

	
	/**
	 * computes the scan range of the scan, note that this is only valid if
	 * the scan range is less or equal to 360 degrees
	 * @return the minimum scan range to cover all given primary angles
	 */
	public double computeScanRange(){
		double out = Double.POSITIVE_INFINITY;
		if (primaryAngles != null){
			return DoubleArrayUtil.minAndMaxOfArray(getPrimaryAngles())[1];
			/*for (int i = 0; i < primaryAngles.length; i++) {
				double min = Double.POSITIVE_INFINITY;
				double max = Double.NEGATIVE_INFINITY;
				for (int j = 0; j < primaryAngles.length; j++) {
					double currAngle = (primaryAngles[j]-primaryAngles[i])%360;
					currAngle = (currAngle < 0) ? currAngle+360 : currAngle;
					min = (currAngle < min) ? currAngle : min;
					max = (currAngle > max) ? currAngle : max;
				}
				if ((max-min) < out){
					out = (max-min);
				}
			}
			*/
		}
		else{
			checkDelta();
			out = (Math.PI + (2.0 * delta));
		}
		return out;
	}


	/**
	 * computes the set of Parker weights for the given projection
	 * @param projNum the projection number
	 * @return the weights
	 */
	public double [] computeParkerWeights1D(int projNum){
		checkDelta();
		double beta = projNum;

		double [] minmax = null;
		double maxRange = (Math.PI + (2.0 * delta));
		if (debug) System.out.println("Angular Max Range: " + maxRange * 180 / Math.PI);
		double [] revan = new double [detectorWidth];
		try {
			double range=0;
			if (primaryAngles != null){
				minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
				//range = (minmax[1] - minmax[0]) * Math.PI / 180.0;
				range = computeScanRange() * Math.PI / 180.0;
				if (debug) {
					System.out.println("Angular Range: " + range * 180 / Math.PI + "\n MaxRange: " + maxRange *180 / Math.PI + "\nProjections: " + primaryAngles.length + " " + numberOfProjections);
				}
				
				// offset has to be computed here
				// otherwise, it is only computed in the configure() method, which may not be called before
				offset = (maxRange - range) /2;
				
				//offset = 0.001;
				//delta *= 0.5;
				//offset -= (0.9 / 180) * Math.PI;

				beta = (primaryAngles[projNum] - minmax[0]+0.0) * Math.PI / 180;
				beta += offset;

				if (debug) System.out.println("delta: " + delta * 180.0 / Math.PI);
				if (debug) System.out.println("Angular Offset: " + offset * 180.0 / Math.PI);
				if (debug) System.out.println("Beta: " + beta * 180.0 / Math.PI);
			} else {
				beta = (0.0 /180 * Math.PI ) +(((projNum + 0.0) / numberOfProjections) + 0.001) / maxRange;
				if (debug) System.out.println("Beta: " + beta * 180.0 / Math.PI);
				range = maxRange;
			}
			
			revan = DoubleArrayUtil.multiply(computeParkerWeights1D(beta), ((range+range/numberOfProjections)/Math.PI));
		} catch (Exception e){
			System.out.println("No Geometry for projection " + projNum + ". Will paint it black.");
		}
		return revan;
	}

	/**
	 * Returns the row weights for the given angle (in radians);
	 * @param beta the angle
	 * @return the row weights
	 */
	public double[] computeParkerWeights1D(double beta){
		double [] revan = new double [detectorWidth];
		for (int i = 0; i< detectorWidth; i++){
			revan[i] = computeParkerWeight(i, beta);
		}
		return revan;
	}

	/**
	 * Converts the detector coordinate x into the angle on the detector and calls computeParkerWeight in angular coorinates 
	 * @param x the row index
	 * @param beta the current angle of the source detector pair.
	 * @return the Parker weight
	 */
	protected double computeParkerWeight(int x, double beta){
		double value = ( ((double)x+0.5) - (double)detectorWidth/2.0 ) * pixelDimensionX;
		double alpha = Math.atan(value / sourceToDetectorDistance);
		return computeParkerWeight(alpha, beta);
	}

	/**
	 * checks whether delta was already computed.
	 */
	protected void checkDelta(){
		if (delta == 0){
			delta = Math.atan(((this.detectorWidth * pixelDimensionX) / 2) / sourceToDetectorDistance)  ;
			if (debug) System.out.println("Delta: " + delta / Math.PI * 180);
		}
	}

	/**
	 * Implements a linear weighting which complies with
	 * 
	 * weight(alpha, beta) + weight (- alpha, beta - Math.PI + (2*alpha)) = 1
	 * (Sum of the weight for opposing rays is 1). 
	 * 
	 * @param alpha the angle of the row element within the interval [-delta,delta]
	 * @param beta the current rotation angle of the source detector pair [0, Math.PI + 2delta]
	 * @param delta the half fan angle;
	 * @return the linear weight
	 */
	public static double linearWeight(double alpha, double beta, double delta){
		double revan = 0;
		double off = -0.1825;
		double slope = off * ((2 * delta + Math.PI  - beta) / ((2 * delta + Math.PI) /2));
		revan = (1-off) - slope * (alpha / delta); 
		return revan;
	}

	/**
	 * Weight for the Begin of the Scan
	 * @param alpha the angle of the row element within the interval [-delta,delta]
	 * @param beta the current rotation angle of the source detector pair [0, Math.PI + 2delta]
	 * @return the begin weight
	 */
	protected double beginWeight(double alpha, double beta){
		return Math.pow(Math.sin((Math.PI/4) * ((beta) / (delta-alpha))),2);
	}

	/**
	 * Weight for the End of the Scan
	 * @param alpha the angle of the row element within the interval [-delta,delta]
	 * @param beta the current rotation angle of the source detector pair [0, Math.PI + 2delta]
	 * @return the end weight
	 */
	protected double endWeight(double alpha, double beta){
		return Math.pow(Math.sin((Math.PI/4) * ((Math.PI + (delta + delta) - beta)/(delta+alpha))), 2);
		//return Math.pow(Math.sin((Math.PI/4) * ((Math.PI + (delta + delta) - beta)/(delta-alpha))), 2);
	}

	/**
	 * Checks whether alpha and beta are within the bounds for the Parker weights at the begin of the scan.
	 * @param alpha the angle of the row element within the interval [-delta,delta]
	 * @param beta the current rotation angle of the source detector pair [0, Math.PI + 2delta]
	 * @return true if the begin condition is fulfilled
	 */
	protected boolean checkBeginCondition(double alpha, double beta){
		return (0 <= beta) && (beta  <= (delta + delta) - (2*alpha) );
	}

	/**
	 * Checks whether alpha and beta are within the bounds for the Parker weights at the end of the scan.
	 * @param alpha the angle of the row element within the interval [-delta,delta]
	 * @param beta the current rotation angle of the source detector pair [0, Math.PI + 2delta]
	 * @return true if the end condition is fulfilled
	 */
	protected boolean checkEndCondition(double alpha, double beta){
		return ((Math.PI - (2*alpha)) <= beta) && (beta <= (Math.PI + delta + delta));
	}


	/**
	 * Computes Parker weight for the given alpha and beta.
	 * @param alpha the angle of the row element within the interval [-delta,delta]
	 * @param beta the current rotation angle of the source detector pair [0, Math.PI + 2delta]
	 * @return the Parker weight
	 */
	protected double computeParkerWeight(double alpha, double beta){
		
		// Shift weights such that they are centered (Important for maxBeta < pi + 2 * gammaM)
		//double[] minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
		//double range = computeScanRange() * Math.PI / 180.0;
		
		checkDelta();
		//beta += (Math.PI+2*delta-range)/2.0;
		/*
		if (beta < 0){
			beta += 2*Math.PI;
		}*/
		if (beta < 0){
			return 0;
		}
		if (beta > Math.PI*2){
			return 0;
		}

		double weight = 1;
		if (beta == 0) beta = CONRAD.SMALL_VALUE;
		if (checkBeginCondition(alpha, beta)){
			// begin of sweep
			double Nweight = beginWeight(alpha, beta);
			weight = Nweight;
		} else {	
			if(checkEndCondition(alpha, beta)){
				// end of sweep
				double Nweight = endWeight(alpha, beta);
				weight = Nweight;
			}
		}
		if (beta > Math.PI + 2 * this.delta) weight = 0;
		return weight;
	}


	/**
	 * Applies the tool to the given image processor.
	 */
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		Grid2D theFilteredProcessor = imageProcessor;
		double [] theWeights = this.computeParkerWeights1D(this.imageIndex);
		if (debug) System.out.println(numberOfProjections);
		if (offset != 0) {
			//theWeights = DoubleArrayUtil.gaussianFilter(theWeights, 20);
		}
		if ((imageIndex < 10) || (imageIndex > this.numberOfProjections -5)) {
			//if (debug) VisualizationUtil.createPlot("Projection " + imageIndex, theWeights).show();
		}
		if (debug) DoubleArrayUtil.saveForVisualization(imageIndex, theWeights);
		//Grid2D output = new Grid2D(theFilteredProcessor.getWidth(),theFilteredProcessor.getHeight());
		for (int j = 0; j < theFilteredProcessor.getHeight(); j++){
			for (int i=0; i < theFilteredProcessor.getWidth() ; i++){
				if (theFilteredProcessor.getWidth() <= theWeights.length) {
					double value = theFilteredProcessor.getPixelValue(i, j) * theWeights[i];
					//output.putPixelValue(i, j, theWeights[i]);
					theFilteredProcessor.putPixelValue(i, j, value);
				} else {
					int offset = (theFilteredProcessor.getWidth() - theWeights.length) / 2;
					if (((i-offset) > 0)&&(i-offset < theWeights.length)){
						double value = theFilteredProcessor.getPixelValue(i, j) * theWeights[i - offset];
						//output.putPixelValue(i, j, theWeights[i-offset]);
						theFilteredProcessor.putPixelValue(i, j, value);
					}
				}
			}
		}
		//IJ.saveAs(new ImagePlus("",ImageUtil.wrapGrid2D(output)),"TIFF", "C:\\Users\\berger\\Desktop\\NormalizedConvolutionTest\\ParkerWeights\\Slice_" + this.getImageIndex() + "_" + Calendar.getInstance().getTimeInMillis() + ".tif");
		return theFilteredProcessor;
	}
	public Grid2D applyToolToImageMirror(Grid2D imageProcessor) {
		Grid2D theFilteredProcessor = imageProcessor;
		double [] theWeights = this.computeParkerWeights1D(this.imageIndex);
		if (debug) System.out.println(numberOfProjections);
		if (offset != 0) {
			//theWeights = DoubleArrayUtil.gaussianFilter(theWeights, 20);
		}
		if ((imageIndex < 10) || (imageIndex > this.numberOfProjections -5)) {
			//if (debug) VisualizationUtil.createPlot("Projection " + imageIndex, theWeights).show();
		}
		if (debug) DoubleArrayUtil.saveForVisualization(imageIndex, theWeights);
		//Grid2D output = new Grid2D(theFilteredProcessor.getWidth(),theFilteredProcessor.getHeight());
		for (int j = 0; j < theFilteredProcessor.getHeight(); j++){
			for (int i=0; i < theFilteredProcessor.getWidth() ; i++){
				if (theFilteredProcessor.getWidth() <= theWeights.length) {
					double value = theFilteredProcessor.getPixelValue(i, j) * theWeights[theFilteredProcessor.getWidth()-1-i];
					//output.putPixelValue(i, j, theWeights[i]);
					theFilteredProcessor.putPixelValue(i, j, value);
				} else {
					int offset = (theFilteredProcessor.getWidth() - theWeights.length) / 2;
					if (((i-offset) > 0)&&(i-offset < theWeights.length)){
						double value = theFilteredProcessor.getPixelValue(i, j) * theWeights[i - offset];
						//output.putPixelValue(i, j, theWeights[i-offset]);
						theFilteredProcessor.putPixelValue(i, j, value);
					}
				}
			}
		}
		//IJ.saveAs(new ImagePlus("",ImageUtil.wrapGrid2D(output)),"TIFF", "C:\\Users\\berger\\Desktop\\NormalizedConvolutionTest\\ParkerWeights\\Slice_" + this.getImageIndex() + "_" + Calendar.getInstance().getTimeInMillis() + ".tif");
		return theFilteredProcessor;
	}
	public void configure() throws Exception {
		Configuration config = Configuration.getGlobalConfiguration(); 
		setConfiguration(config);
		//double [] minmax = null;
		checkDelta();
		double maxRange = (Math.PI + (2 * delta));
		if (primaryAngles != null){
			//minmax = DoubleArrayUtil.minAndMaxOfArray(primaryAngles);
			double range = computeScanRange() * Math.PI / 180;
			offset = (maxRange - range) /2;
			if (debug) System.out.println("delta: " + delta * 180 / Math.PI);
			if (debug) System.out.println("Angular Offset: " + offset * 180 / Math.PI + " " + maxRange + " " + range);
		}

		//offset = UserUtil.queryDouble("Offset for Parker weights: ", offset);
		setNumberOfProjections(config.getGeometry().getPrimaryAngles().length);
		if (this.numberOfProjections == 0){
			throw new Exception("Number of projections not known");
		}
		setConfigured(true);
	}

	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Parker82-OSS,\n" +
		"  author = {{Parker}, D. L.},\n" +
		"  title = \"{{Optimal short scan convolution reconstruction for fanbeam CT}}\",\n" +
		"  journal = {Medical Physics},\n" +
		"  year = 1982,\n" +
		"  volume = 9,\n"+
		"  number = 2,\n" +
		"  pages = {254-257}\n" +
		"}";
		return bibtex;
	}

	public String getMedlineCitation() {
		String medline = "Parker DL. Optimal short scan convolution reconstruction for fanbeam CT. Med Phys. 1982 Mar-Apr;9(2):254-7.";
		return medline;
	}

	/**
	 * Is used to compensate ambiguities caused by double rays. Is not device dependent, but scan geometry dependent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}

}

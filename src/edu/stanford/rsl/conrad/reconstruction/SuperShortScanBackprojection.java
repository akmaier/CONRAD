package edu.stanford.rsl.conrad.reconstruction;


import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.reconstruction.ReconstructionFilter;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import ij.IJ;

@Deprecated
public abstract class SuperShortScanBackprojection extends ReconstructionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8744415315605892588L;
	protected double [] primaryAngles;
	protected double SID;
	protected double width =-1;
	protected double height = -1;
	protected double detectorElementSizeX;
	protected double detectorElementSizeY;
	private boolean closed = false;

	protected ReconstructionFilter projector;
	protected ImageGridBuffer imageBuffer;
	protected Grid2D[] projections;


	@Override
	public void configure() throws Exception{
		projectionVolume = null;
		Configuration config = Configuration.getGlobalConfiguration();
		SID = config.getGeometry().getSourceToDetectorDistance();
		detectorElementSizeX = config.getGeometry().getPixelDimensionX();
		detectorElementSizeY = config.getGeometry().getPixelDimensionY();
		//maxProjections = config.getGeometry().getProjectionStackSize();
		//projections = new ImageProcessor[maxProjections];
		//ReconstructionFilter [] projectors = BufferedProjectionSink.getBackprojectors();
		//projector = (ReconstructionFilter) UserUtil.chooseObject("Select Backprojector", "Backprojector Selection", projectors, projector);
		projector.configure();
		configured = true;
	}


	@Override
	public String getName(){
		return "Super Short Scan Backprojector";
	}

	@Override
	public synchronized void process(Grid2D img, int index) throws Exception{
		if (imageBuffer == null){
			imageBuffer = new ImageGridBuffer();
		}
		imageBuffer.add(img, index);
	}

	public void close(){
		//if (true) throw new RuntimeException("closing" + closed);
		if (!closed) {
			projections = imageBuffer.toArray();
			int maxProjections = imageBuffer.size();
			imageBuffer = null;
			// Backproject using selected algorithm
			for(int i=0; i < maxProjections; i++){
				IJ.showStatus("Backprojecting");
				IJ.showProgress(((float)i)/maxProjections);
				try {
					
					projector.process(projections[i], i);
				} catch (Exception e){
					e.printStackTrace();
				}
			}
			IJ.showProgress(1.0);
			projector.close();
			//projectionVolume = projector.getResult();
			//for (int n = 1; n <=projectionVolume.getNSlices(); n++){
			//	projectionVolume.getStack().getProcessor(n).multiply(-1.0);
			//}
			//projectionVolume.show();
			closed = true;
		}
	}
	
/*
	public void close_old() throws Exception{
		if (!closed) {
			projections = imageBuffer.toArray();
			int maxProjections = imageBuffer.size();
			imageBuffer = null;
			// precompute weights
			double [] uWeights = new double[projections[0].getWidth()];
			for (int i=0; i<uWeights.length;i++){
				double pos = (i-(uWeights.length/2)) * detectorElementSizeX;
				uWeights[i]= ((Math.pow(pos,2) + Math.pow(SID, 2)) / SID);
			}
			double [][] vWeights = new double[projections[0].getWidth()][projections[0].getHeight()];
			for (int j=0; j<vWeights[0].length;j++){
				for (int i=0; i<vWeights.length;i++){
					double posU = (i-(vWeights.length/2)) * detectorElementSizeX;
					double posV = (j-(vWeights[0].length/2)) * detectorElementSizeY;
					vWeights[i][j]= (posU * posV) / SID;
				}
			}
			// perform triplet-wise derivative;
			ImageProcessor lastProjection = projections[0].duplicate();
			for (int i = 1; i< maxProjections-1; i++){
				IJ.showStatus("Computing lambda derivative");
				IJ.showProgress(((float)i)/maxProjections);
				ImageProcessor projectionBasedDerivative = computeUVDerivative(i, uWeights, vWeights);
				ImageProcessor lambdaDerivative = lastProjection;
				//if (i == 15) VisualizationUtil.showImageProcessor(lambdaDerivative.duplicate());
				//lambdaDerivative.multiply(-1.0);
				//if (i == 15) VisualizationUtil.showImageProcessor(lambdaDerivative.duplicate());
				ImageUtil.subtractProcessors(lambdaDerivative, projections[i]);
				//if (i == 15) VisualizationUtil.showImageProcessor(lambdaDerivative.duplicate());
				//lambdaDerivative.multiply(-1.0);
				//if (i == 15) VisualizationUtil.showImageProcessor(lambdaDerivative.duplicate());
				ImageUtil.addProcessors(lambdaDerivative, projectionBasedDerivative);
				lastProjection = projections[i];
				projections[i] = lambdaDerivative;
			}
			// we assume derivative 0 in first and last projection in lambda direction.
			projections[0]=computeUVDerivative(0, uWeights, vWeights);
			projections[maxProjections-1]=computeUVDerivative(maxProjections-1, uWeights, vWeights);
			// Backproject using selected algorithm
			for(int i=0; i < maxProjections; i++){
				IJ.showStatus("Backprojecting");
				IJ.showProgress(((float)i)/maxProjections);
				projector.process(projections[i], i);
			}
			IJ.showProgress(1.0);

			closed = true;
		}
	}

	private ImageProcessor computeUVDerivative(int index, double [] uWeights, double [][] vWeights){
		float [] kernel = {-1, 0, 1};
		ImageProcessor revan = new FloatProcessor(projections[index].getWidth(), projections[index].getHeight());
		// derivative in row direction
		ImageProcessor uDev = projections[index].duplicate();
		uDev.convolve(kernel, 3, 1);
		// derivative in column direction
		ImageProcessor vDev = projections[index].duplicate();
		vDev.convolve(kernel, 1, 3);
		// weight and add to result.
		for(int i =0; i<uDev.getWidth(); i++){
			for (int j=0; j< uDev.getHeight(); j++){
				double u = uDev.getPixelValue(i,j);
				u *= uWeights[i];
				double v = vDev.getPixelValue(i, j);
				v *= vWeights[i][j];
				revan.putPixelValue(i, j, v+u);
			}
		}
		return revan;
	}
*/
	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		primaryAngles = null;
		projections = null;
		projector = null;
		configured = false;
		closed = false;
		imageBuffer = null;
	}

	@Override
	public String getBibtexCitation(){
		return "see medline";
	}

	@Override
	public String getMedlineCitation(){
		return "Yu H, Wang G. Feldkamp-type VOI reconstruction from super-short-scan cone-beam data. Med Phys 31(6):1357-62. 2004.";
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
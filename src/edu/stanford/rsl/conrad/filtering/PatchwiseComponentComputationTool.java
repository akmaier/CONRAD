package edu.stanford.rsl.conrad.filtering;

import org.fastica.FastICA;
import org.fastica.FastICAException;

import weka.core.matrix.Matrix;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class PatchwiseComponentComputationTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3740372509956056879L;
	private String operation = null;
	private int patchSize = 160;
	private int patchSkip = 80;
	public static final String SVD = " SVD ";
	public static final String PCA = " PCA ";
	public static final String ICA = " ICA ";
	public static final String Correlation = " Local Correlation ";


	@Override
	public void configure() throws Exception {
		String [] operations = {PCA, SVD, ICA, Correlation};
		operation = (String) UserUtil.chooseObject("Select operation: ", "Operation Selection", operations, operation);
		patchSize = UserUtil.queryInt("Enter Patch Size: ", patchSize);
		patchSkip = UserUtil.queryInt("Enter skip after patch: ", patchSkip);
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		if (!configured) return "Patchwise Component Computation Tool";
		else return "Patchwise Component Computation Tool " + operation + " " + patchSize + "x"+patchSize + " Skip " + patchSkip;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		PatchwiseComponentComputationTool clone = new PatchwiseComponentComputationTool();
		clone.operation = operation;
		clone.patchSkip = patchSkip;
		clone.patchSize = patchSize;
		return clone;
	}

	public String getOperation() {
		return operation;
	}

	public void setOperation(String operation) {
		this.operation = operation;
	}

	public int getPatchSize() {
		return patchSize;
	}

	public void setPatchSize(int patchSize) {
		this.patchSize = patchSize;
	}

	public int getPatchSkip() {
		return patchSkip;
	}

	public void setPatchSkip(int patchSkip) {
		this.patchSkip = patchSkip;
	}

	/**
	 * computes the components of the signals with either PCA, SVD, or ICA.
	 * @param signals the signals
	 * @param numChannels the number of channels
	 * @param operation either PCA, SVD, or ICA.
	 * @return the components
	 * @throws FastICAException
	 */
	public static double [][] getComponents(double [][] signals, int numChannels, String operation) throws Exception{
		double [] sum = new double [signals[0].length];
		for (int i=0;i<signals[0].length;i++){
			for (int k=0;k<numChannels;k++){
				sum[i] += signals[k][i];
			}
		}
		double [][] vectors = null;

		if (operation.equals(ICA)){
			FastICA ica = new FastICA(signals, numChannels);
			vectors = ica.getICVectors();
		}
		if (operation.equals(PCA)||operation.equals(Correlation)){
			org.fastica.PCA pca = new org.fastica.PCA(signals);
			vectors = org.fastica.math.Matrix.mult(pca.getEigenVectors(), pca.getVectorsZeroMean());
			vectors[vectors.length-1] = sum;
			for (int k=0;k<numChannels;k++){
				//System.out.println("Eigen Value "+k+" "+pca.getEigenValues()[k]);
				//vectors = org.fastica.math.Matrix.scale(vectors, pca.getEigenValues()[k]);
			}
		}
		if(operation.equals(SVD)){
			weka.core.matrix.SingularValueDecomposition svd = new weka.core.matrix.SingularValueDecomposition(new Matrix(signals).transpose());
			vectors=svd.getU().transpose().getArray();
			for (int k=0;k<numChannels;k++){
				//System.out.println("Singular Value "+k+" "+svd.getSingularValues()[k]);
				//vectors = org.fastica.math.Matrix.scale(vectors, svd.getSingularValues()[k]);
			}
		}
		SimpleMatrix inverseFactors = new SimpleMatrix(vectors).transposed().inverse(InversionType.INVERT_SVD);

		SimpleVector sumVec = new SimpleVector(sum);
		SimpleVector weights = SimpleOperators.multiply(inverseFactors, sumVec);
		double [][] vectorsReturn = new double [vectors.length+2][];
		vectorsReturn [vectors.length] = sum;
		for (int k=0;k<numChannels;k++){
			vectorsReturn[k] = org.fastica.math.Vector.scale(weights.getElement(k), vectors[k]);
		}
		sum = new double [signals[0].length];
		for (int i=0;i<signals[0].length;i++){
			for (int k=0;k<numChannels;k++){
				sum[i] += vectorsReturn[k][i];
			}
		}
		if (operation.equals(Correlation)){
			double correlation = DoubleArrayUtil.correlateDoubleArrays(signals[0], signals[1]);
			for (int k=0;k<numChannels;k++){
				for (int i=0;i<signals[0].length;i++){
					vectorsReturn[k][i] = correlation;
				}
			}
		}
		vectorsReturn [vectors.length+1] = sum;
		return vectorsReturn;
	}

	/**
	 * Extracts a patch from a MultiChannelGrid at position (x,y). The patch position is indicated by the top left corner.
	 * @param multiGrid the multiChannelGrid
	 * @param x the top left corner x
	 * @param y the top left corner y
	 * @return
	 */
	public double [][] getSignals (MultiChannelGrid2D multiGrid, int x, int y){
		int numChannels = multiGrid.getNumberOfChannels();
		double [][] signals = new double [numChannels][patchSize*patchSize];
		for (int j=0; j < patchSize; j++){
			for (int i=0; i < patchSize; i++){
				int xi = x+i;
				if (xi < multiGrid.getWidth()) for (int k=0;k<numChannels;k++){
					int yj = y + j;
					if (yj < multiGrid.getHeight()) {
						signals[k][(j*patchSize)+i] = multiGrid.getChannel(k).getPixelValue(xi, yj);
						if (! Double.isFinite(signals[k][(j*patchSize)+i])) signals[k][(j*patchSize)+i] = 0;
					}
				}
			}
		}
		return signals;
	}

	/**
	 * Writes a patch to a MultiChannelGrid at position (x,y). The patch position is indicated by the top left corner.
	 * @param multiGrid the multiChannelGrid
	 * @param x the top left corner x
	 * @param y the top left corner y
	 */
	public void putSignals (MultiChannelGrid2D multiGrid, double [][] signals, int x, int y, boolean swap){
		int numChannels = multiGrid.getNumberOfChannels()-2;
		for (int j=0; j < patchSize; j++){
			for (int i=0; i < patchSize; i++){
				int xi = x+i;
				if (xi < multiGrid.getWidth()) 
					for (int k=0;k<numChannels;k++){
						int k2 = k;
						if (swap) k2 = (k2+1)%numChannels;
						int yj = y + j;
						if (yj < multiGrid.getHeight())
							//multiGrid.getChannel(k).putPixelValue(xi, yj, signals[k2][(j*patchSize)+i]);	
							multiGrid.getChannel(k).putPixelValue(xi, yj, signals[k2][(j*patchSize)+i]+multiGrid.getChannel(k).getPixelValue(xi, yj));
					}
				for (int k=numChannels;k<numChannels+2;k++){
					int k2 = k;
					int yj = y + j;
					if (yj < multiGrid.getHeight())
						multiGrid.getChannel(k).putPixelValue(xi, yj, signals[k2][(j*patchSize)+i]);	
				}
			}
		}
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		if (imageProcessor instanceof MultiChannelGrid2D){
			MultiChannelGrid2D multi = (MultiChannelGrid2D) imageProcessor;
			MultiChannelGrid2D out = new MultiChannelGrid2D(multi.getWidth(), multi.getHeight(), multi.getNumberOfChannels()+2);
			int y =0;
			while (y<multi.getHeight()) {
				int x =0;
				while (x<multi.getWidth()) {
					try{
						double [][] signals = getSignals(multi, x, y);
						double [][] components = getComponents(signals, multi.getNumberOfChannels(), operation);
						double mean1 = DoubleArrayUtil.computeMean(components[0]);
						double mean2 = DoubleArrayUtil.computeMean(components[1]);
						double stabw1 = DoubleArrayUtil.computeStddev(components[0], mean1);
						double stabw2 = DoubleArrayUtil.computeStddev(components[1], mean2);
						//System.out.println("Means: " + mean1 + " "+ mean2);
						//System.out.println("Stabw: " + stabw1 + " "+ stabw2);
						putSignals(out, components, x, y, stabw1< stabw2);
					}
					catch (Exception e){
						e.printStackTrace();
					}
					x+= patchSkip;
				}
				y += patchSkip;
			}

			return out;
		} else return imageProcessor;
	}

}

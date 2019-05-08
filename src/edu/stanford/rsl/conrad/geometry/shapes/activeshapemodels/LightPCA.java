/*
 * Copyright (C) 2017 Tobias Geimer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels;

import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.ActiveShapeModel;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.PCA;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.DataMatrix;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.ImageJ;
import ij.gui.Plot;

/**
 * This is a memory light implementation of Principal Component Analysis.
 * 
 * In contrast to {@link PCA} the input data matrix is not stored as a member but discarded after processing.
 * All other PCA-related properties are accessible through the class members after run().
 * 
 * Additionally, this implementation provides methods adapted from {@link ActiveShapeModel} for out-of-sample
 * extension of unseen observations.
 * 
 * Also see Jolliffe, Ian. Principal component analysis. John Wiley & Sons, Ltd, 2005.
 * 
 * @version 2017-12-21;
 * Note that now both {@link LightPCA} and {@link PCA} formulate pc weights with respect to the standard deviation
 * (i.e. square root of the variance).
 * 
 * @author Tobias Geimer
 */
public class LightPCA {
	// ----------------------------------------------------------------------------------------------------
	// MEMBERS
	// ----------------------------------------------------------------------------------------------------

	// ---------------
	// Data Properties
	/**
	 * Data-sets typically consist of point-clouds or meshes. This variable stores 
	 * the dimension of the vertices, as the data is stored in a single column.
	 */
	public int dimension = 0;
	
	/**
	 * Data-sets typically consist of point-clouds or meshes. This variable stores 
	 * the number of vertices, as the data is stored in a single column.
	 */
	public int numVertices;
	
	/**
	 * Number of points in a data-set. Can be calculated using the number of 
	 * vertices and their dimension using multiplication. Corresponds to the number of rows.
	 */
	public int numPoints;
	
	/**
	 * Number of samples in the data matrix, corresponds to the number of columns.
	 */
	public int numSamples;
	/**
	 * Consensus of the data matrix.
	 */
	public SimpleMatrix consensus;
	
	
	// ---------------
	// PCA Properties
	/**
	 * Number of principal components.
	 */
	public int numComponents;
	
	/**
	 * Matrix containing the Eigenvectors of the covariance matrix after singular value decomposition.
	 */
	public SimpleMatrix eigenVectors;
			
	/**
	 * Array containing the eigenvalues of the covariance matrix after singular value decomposition.
	 * This is equivalent to the variance of the dataset.
	 */
	public double[] eigenValues;
	
	/**
	 * Array containing the square-roots of the eigenvalues of the covariance matrix.
	 * This is equivalent to the standard deviation of the data and is used to scale the feature weights.
	 * (E.g. most Active Shape Models limit their allowed values to [-3;3] times the standard deviation. 
	 */
	public double[] standardDeviation;
	
	/**
	 * Threshold at which principal components will be omitted if a variation of this value is reached.
	 * Either this or the actual number of components needs to be provided.
	 */
	public double variationThreshold = 1;
	
	/**
	 * Feature weights (i.e. pc scores) for the data matrix.
	 */
	private SimpleMatrix features;
	
	// ---------------
	// Debug	
	public boolean DEBUG = false;
	public boolean SHOW_EIGEN_VALUES = false;
	// Sanity
	private boolean initialized = false;
	
	
	// ----------------------------------------------------------------------------------------------------
	// CONSTRUCTOR
	// ----------------------------------------------------------------------------------------------------
	/**
	 * Constructs a {@link LightPCA} object and initializes the data array.
	 * Run()-method is called immediately using a number of components, that are needed to reach
	 * variationThreshold (in [0;1]) explained variance. 
	 * Does not support scaling of the DataMatrix currently.
	 * @param data The data array to be analyzed.
	 * @param variationThreshold The amount of variance of the data to be explained by the model.
	 */
	public LightPCA( DataMatrix data, double variationThreshold) {
		this.dimension = data.dimension;
		this.numVertices = data.getRows() / data.dimension;
		this.numPoints = data.getRows();
		this.numSamples = data.getCols();
		this.consensus = data.consensus;
		this.variationThreshold= variationThreshold;
		this.initialized = true;
		this.run(data);
	}
	
	/**
	 * Constructs a {@link LightPCA} object and initializes the data array.
	 * Run()-method is called immediately using a number of principal components
	 * according to numComponents.
	 * @param data The data array to be analyzed.
	 * @param numComponents The number of principal components to be used.
	 */
	public LightPCA( DataMatrix data, int numComponents ) {
		this.dimension = data.dimension;
		this.numVertices = data.getRows() / data.dimension;
		this.numPoints = data.getRows();
		this.numSamples = data.getCols();
		this.numComponents = numComponents;
		this.consensus = data.consensus;
		this.initialized = true;
		this.run(data, this.numComponents);
	}

	/**
	 * Default constructor, needs to be followed up by
	 * defining the variationThreshold and a run(DataMatrix data) call
	 * OR
	 * by run(DataMatrix data, int numComponents)
	 */
	public LightPCA( ) {

	}
	/**
	 * Initialize the data properties using information from the data matrix to be processed.
	 * @param data
	 */
	public void init(DataMatrix data) {
		this.dimension = data.dimension;
		this.numVertices = data.getRows() / data.dimension;
		this.numPoints = data.getRows();
		this.numSamples = data.getCols();
		this.consensus = data.consensus;
		this.initialized = true;
	}

	// ----------------------------------------------------------------------------------------------------
	// METHODS
	// ----------------------------------------------------------------------------------------------------
	public void run(DataMatrix data) {
		if( !this.initialized ) this.init(data);
		
		if(DEBUG) System.out.println("Starting principal component analysis on " + numSamples + " data-sets.");
		
		// Make data matrix be zero-centered.
		data = subtractConsensus(data);
		
		DecompositionSVD svd = new DecompositionSVD(data);
		
		// The eigenvalues sigma_i of the covariance matrix are given as the square of
		// the signular values s_i of the data amtrix, scaled with the number of samples.
		// sigma_i = s_i^2 / (#samples-1)
		double[] eigenValues = new double[svd.getSingularValues().length];
		for( int i = 0; i < eigenValues.length; i++ ) {
			eigenValues[i] = Math.pow(svd.getSingularValues()[i],2)/(this.numSamples-1);
		}
		
		plot(eigenValues);
		
		// Determine the number of components needed to reach variationThreshold.
		this.numComponents = getPrincipalModesOfVariation(eigenValues);
		
		// Set the first numComponents eigenValues, eigenVectors and standardDeviations.
		reduceDimensionality(eigenValues, normalizeColumns(svd.getU()));
		
		// Compute the pc scores for the data matrix
		double[][] weights = new double[data.getCols()][this.numComponents];
		for(int i = 0; i < data.getCols(); i++ ) {
			weights[i] = this.projectTrainingShape(data.getCol(i),i);
		}

		// Wrap buffer into SimpleMatrix and transpose to have
		// weights for each sample in columns again.
		this.features = (new SimpleMatrix(weights)).transposed();
						
		// Rebuilt data matrix because PCA subtracted the consensus.
		this.addConsensus(data);		
	}

	public void run(DataMatrix data, int numComponents) {
		if(!this.initialized) this.init(data);
		
		if(DEBUG) System.out.println("Starting principal component analysis on " + numSamples + " data-sets.");
		
		// Make data matrix be zero-centered.
		data = subtractConsensus(data);
				
		DecompositionSVD svd = new DecompositionSVD(data);
		
		// The eigenvalues sigma_i of the covariance matrix are given as the square of
		// the singular values s_i of the data matrix, scaled with the number of samples.
		// sigma_i = s_i^2 / (#samples-1)
		double[] eigenVals = new double[svd.getSingularValues().length];
		for( int i = 0; i < eigenVals.length; i++ ) {
			eigenVals[i] = Math.pow(svd.getSingularValues()[i],2)/(this.numSamples-1);
		}
		
		plot(eigenVals);
		
		// Set the number of components.
		this.numComponents = numComponents;
		
		// Set the first numComponents eigenValues, eigenVectors and standardDeviations.
		reduceDimensionality(eigenVals, normalizeColumns(svd.getU()));
		
		// Compute the pc scores for the data matrix
		double[][] weights = new double[data.getCols()][this.numComponents];
		for(int i = 0; i < data.getCols(); i++ ) {
			weights[i] = this.projectTrainingShape(data.getCol(i),i);
		}

		// Wrap buffer into SimpleMatrix and transpose to have
		// weights for each sample in columns again.
		this.features = (new SimpleMatrix(weights)).transposed();
						
		// Rebuilt data matrix because PCA subtracted the consensus.
		this.addConsensus(data);		
	}
	
	// ----------------------------------------------------------------------------------------------------
	// PCA Properties
	/**
	 * Calculates the principal components that are necessary to reach the threshold of variation set in the class member.
	 * If the plot flag has been set to true, the variation analysis will be plotted as function of the principal components.
	 * @param ev The eigenvalues of the covariance matrix.
	 * @return The threshold index for the principal components.
	 */
	public int getPrincipalModesOfVariation(double[] ev){
		double sum = 0;
		for(int i = 0; i < ev.length; i++){
			sum += ev[i];
		}
		double[] var = new double[ev.length];
		
		var[0] = ev[0] / sum;
		for(int i = 1; i < ev.length; i++){
			var[i] = var[i-1] + ev[i] / sum;
		}
		
		int i = 0;
		while(var[i] < variationThreshold && i<ev.length-1){
			i++;
		}
		i++;
		
		if(SHOW_EIGEN_VALUES){
			Plot plot = VisualizationUtil.createPlot(var, "Variation as function of principal component", "Principal Component", "Variation");
			plot.show();
		}
		
		return (i<ev.length)?i:ev.length;
	}
	
	/**
	 * Allocates and sets the principal components and the corresponding variation values. Is used for dimensionality reduction after 
	 * the amount of principal components needed has been determined.
	 * @param ev The eigenvalues of the covariance matrix.
	 * @param pc The principal components (i.e. eigenvectors)
	 */
	private void reduceDimensionality(double[] ev, SimpleMatrix pc){
		this.eigenValues = new double[numComponents];
		this.standardDeviation = new double[numComponents];
		this.eigenVectors = new SimpleMatrix(numPoints, numComponents);
	
		for(int i = 0; i < numComponents; i++){
			this.eigenVectors.setColValue(i, pc.getCol(i));
			this.eigenValues[i] = ev[i];
			this.standardDeviation[i] = Math.sqrt(ev[i]);
		}
	}
	
	/**
	 * Normalizes the columns of a matrix.
	 * @param m The matrix whose columns will be normalized.
	 * @return A matrix with normalized column vectors.
	 */
	private SimpleMatrix normalizeColumns(SimpleMatrix m){
		SimpleMatrix norm = new SimpleMatrix(m.getRows(), m.getCols());
				
		for(int j = 0; j < m.getCols(); j++){
			double s = 0;
			
			for(int i = 0; i < m.getRows(); i++){
				s += Math.pow(m.getElement(i, j),2);
			}
			
			s = Math.sqrt(s);
			norm.setColValue(j, m.getCol(j).dividedBy(s));
		}
		
		return norm;
	}
	
	/** 
	 * Projects training shape in the form of (@link SimpleVector) onto the principal components.
	 * Assumes that consensus is already subtracted and the data matrix has been scaled accordingly.
	 * @param num
	 * @return
	 */
	private double[] projectTrainingShape(SimpleVector shape, int num){
		assert(this.eigenValues != null) : new Exception("Run analysis first.");
		
		double[] weights = new double[numComponents];
		for(int i = 0; i < numComponents; i++){
			SimpleVector comp = eigenVectors.getCol(i);
			double val = SimpleOperators.multiplyInnerProd(shape, comp);
			shape.subtract(comp.multipliedBy(val));
			weights[i] = val/this.standardDeviation[i];
		}		
		double error = shape.normL2()/shape.getLen();
		if(DEBUG) System.out.println("Mapping error for " + num + ": " + error);
		return weights;
	}
	
	/**
	 * Subtracts the consensus from the data-sets and re-scales the data-sets.
	 * @param mat The data-sets.
	 * @return The data-set after consensus subtraction.
	 */
	private DataMatrix subtractConsensus(DataMatrix mat){
		for(int k = 0; k < numSamples; k++){
//			float factor = mat.scaling.get(k);
			for(int i = 0; i < numVertices; i++){
				SimpleVector row = mat.consensus.getRow(i);
				for(int j = 0; j < dimension; j++){
//					mat.multiplyElementBy(i * dimension + j, k, factor);
					mat.subtractFromElement(i * dimension + j, k, row.getElement(j));
				}
			}
		}
		return mat;
	}
	
	/**
	 * Since PCA subtracted the consensus and applied scaling, the training DataMatrix needs to be
	 * rebuilt in order to not compromise it for further processing due to in-place processing.
	 * @param data
	 */
	private void addConsensus(DataMatrix mat){
		for(int k = 0; k < numSamples; k++){
//			float factor = mat.scaling.get(k);
			for(int i = 0; i < numVertices; i++){
				SimpleVector row = mat.consensus.getRow(i);
				for(int j = 0; j < dimension; j++){
					mat.addToElement(i* dimension + j, k, row.getElement(j));
//					mat.multiplyElementBy(i * dimension + j, k, 1/factor);
				}
			}
		}
	}
	
	// ----------------------------------------------------------------------------------------------------
	// Getter Methods
	/**
	 * Getter for feature weights of training samples.
	 * @return The pca scores for the training samples.
	 */
	public SimpleMatrix getFeatureWeights() {
		return this.features;
	}

	/**
	 * Getter for feature weights of the sample at idx within the training samples.
	 * @param idx The index of the sample of interest.
	 * @return The pca scores for the sample at idx.
	 */
	public double[] getFeaturesAtIndex( int idx ) {
		return this.features.getCol(idx).copyAsDoubleArray();
	}


	// ----------------------------------------------------------------------------------------------------
	// Model Methods
	/**
	 * Project a deformation field instance onto the principle components of the active shape model.
	 * @param mat the deformation field to be expressed in terms of the model
	 * @return an array of feature weights
	 */
	public double[] outOfSampleExtension(SimpleMatrix mat){
		// substract the data consensus
		SimpleMatrix cenMat = SimpleOperators.subtract(mat, this.consensus);
		SimpleVector shape = this.linearizeSimpleMatrix(cenMat);

		double[] weights = new double[this.numComponents];

		// multiply with eigenvectors of the shape
		for(int i = 0; i < this.numComponents; i++){
			SimpleVector comp = this.eigenVectors.getCol(i);
			double val = SimpleOperators.multiplyInnerProd(shape, comp);
			shape.subtract(comp.multipliedBy(val));
			// Weights are normed with the standard deviation along that component.
			weights[i] = val/this.standardDeviation[i];
		}		
		double error = shape.normL2()/shape.getLen();
		if(DEBUG) System.out.println("Mapping error: " + error);
		return weights;
	}


	/**
	 * Reconstructs a {@link SimpleMatrix} from feature weights
	 * by a linear combination of principal components and addition of the data consensus.
	 * 
	 * @param weights double array of pca scores
	 * @return reconstructed deformation field
	 */
	public SimpleMatrix applyWeight(double[] weights) {
		assert(weights.length == this.eigenVectors.getCols()) : new Exception("Weights don't match the size of the score matrix.");
		SimpleVector col = new SimpleVector(numPoints);
		
		// Linear combination of eigenvectors.
		for(int i = 0; i < weights.length; i++){
			col.add(this.eigenVectors.getCol(i).multipliedBy(weights[i] * this.standardDeviation[i]));
		}
		
		// Add the consensus.
		for(int i = 0; i < numVertices; i++){
			SimpleVector row = this.consensus.getRow(i);
			for(int j = 0; j < dimension; j++){
				col.addToElement(i * dimension + j, row.getElement(j));
			}
		}
		
		return this.toPointlikeMatrix(col,this.dimension);
	}
	
	// ----------------------------------------------------------------------------------------------------
	// Helper
	/**
	 * Plots the data in the array over its array index.
	 * @param data The data to be plotted.
	 */
	private void plot(double[] data){
		if(SHOW_EIGEN_VALUES){
			new ImageJ();
			Plot plot = VisualizationUtil.createPlot(data, "Eigenvalues of covariance matrix", "Eigenvalue", "Magnitude");
			plot.show();
		}
	}
	
	/**
	 * Helper method to linearize a SimpleMatrix to a SimpleVector.
	 * The structure inside the vector will be as follows: i-th row p_i with elements (x_ij)
	 * will be written to M_(i*j+j). 	
	 * @param input The SimpleMatrix to be linearized.
	 * @return SimpleVector The linearized data.
	 */
	private SimpleVector linearizeSimpleMatrix( SimpleMatrix input ) {
		int sz = input.getCols()*input.getRows();
		SimpleVector vec = new SimpleVector(sz);
		
		// iterate over rows
		for( int r = 0; r < input.getRows(); r++) {
			// index offset
			int c = r * input.getCols(); 
			// fill matrix column into SimpleVector at given offset
			vec.setSubVecValue(c, input.getRow(r));
		}
		
		return vec;
	}
	
	/**
	 * Reshapes a vector to a matrix using the assumption, that the vector's entries are points of a certain dimension.
	 * @param vec The vector containing the data.
	 * @param dim The dimension of each vertex.
	 * @return	The reshaped data as matrix.
	 */
	private SimpleMatrix toPointlikeMatrix(SimpleVector vec, int dim){
		int numVertices = vec.getLen() / dim;
		SimpleMatrix mat = new SimpleMatrix(numVertices, dim);
		
		for(int i = 0; i < numVertices; i++){
				mat.setRowValue(i, vec.getSubVec(i * dim, dim));
		}
		
		return mat;
	}
	
	/**
	 * Without retraining, further reduces the dimensionality of the PC basis,
	 * by removing all but the first numComp eigenvalues and eigenvectors.
	 * The new number of principal components needs to be smaller than current {@link this.numComponents}. Throws an {@link IllegalArgumentException} otherwise.
	 * 
	 * @param numComp number of principal components
	 */
	public void reduceDimensionalityWithoutRetraining(int numComp) {
		if(this.numComponents > numComp) {
			throw new IllegalArgumentException("Can only ever decrease number of components, which is currently " + this.numComponents);
		}
		
		this.numComponents = numComp;
		this.reduceDimensionality(this.eigenValues, this.eigenVectors);
	}
}

/*
 * Copyright (C) 2010-2017 Tobias Geimer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
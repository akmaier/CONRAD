/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels;

import java.util.ArrayList;

import ij.ImageJ;
import ij.gui.Plot;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.DataMatrix;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * This class performs Principal Component Analysis (PCA) on a data-set. The data-set can be composed of scalar values or 
 * of any dimension, e.g. vertices of a surface mesh representing shape. The columns of the data-set are treated as random 
 * variables, hence one sample needs to be stored in one column only, no matter what dimensionality. 
 * The implementation here calculates the Eigen-Analysis of the covariance matrix using a singular value decomposition. The 
 * Eigen-Values and -Vectors of the covariance matrix will be accessible through the class members.
 * The implementation assumes, that the dataset has been subject to Generalized Procrustes Alignment. If an implementation of GPA 
 * other than the one provided here is used, modifications to PCA (e.g. re-scaling and consensus subtraction) might not be necessary.
 * Jolliffe, Ian. Principal component analysis. John Wiley & Sons, Ltd, 2005.
 * @author Mathias Unberath
 *
 */
public class PCA {
	
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
	 * The matrix containing the data-sets to be analyzed.
	 */
	public DataMatrix data;
	
	/**
	 * Connectivity information when dealing with a mesh.
	 */
	public SimpleMatrix connectivity;
	
	/**
	 * Matrix containing the Eigenvectors of the covariance matrix after singular value decomposition.
	 */
	public SimpleMatrix eigenVectors;
			
	/**
	 * Array containing the eigenvalues of the covariance matrix after singular value decomposition.
	 */
	public double[] eigenValues;
	
	/**
	 * Number of principal Components needed to reach variation threshold.
	 */
	public int numComponents;
	/**
	 * Threshold at which principal components will be omitted if a variation of this value is reached.
	 */
	public double variationThreshold = 1;
	
	public boolean PLOT_SINGULAR_VALUES = false;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	/**
	 * Constructs an empty PCA object. Variables need to be initialized before analysis can be performed.
	 * TODO implement init method for this constructor
	 */
	public PCA(){	
	}
	
	/**
	 * Constructs a PCA object and initializes the data array and count variables.
	 * @param data The data array to be analyzed.
	 */
	public PCA(DataMatrix data){
		this.numPoints = data.getRows();
		this.numSamples = data.getCols();
		this.dimension = data.dimension;		
		this.numVertices = numPoints / dimension;
		data = scaleConsensus(data);
		this.data = upscaleAndSubtractConsensus(data);
		
		
		if(data.HAS_CONNECTIVITY){
			this.connectivity = data.connectivity;
		}
	}
	
	/**
	 * Constructs a PCA object and initializes the data array.
	 * Due to the lacking information about scaling factors and consensus object, this constructor is not to be 
	 * used for statistical shape model generation after generalized procrustes analysis.
	 * @param data The data array to be analyzed.
	 * @param dim The dimension of the data points.
	 */
	public PCA(SimpleMatrix data, int dim){
		this.numPoints = data.getRows();
		this.numSamples = data.getCols();
		
		this.dimension = dim;
		
		this.numVertices = numPoints / dimension;
		
		DataMatrix datam = new DataMatrix();
		datam.setDimensions(data.getRows(), dim, data.getCols());
		datam.add(data);
		datam.scaling = new ArrayList<Float>();
		for(int i = 0; i < data.getCols(); i++){
			datam.scaling.add(1f);
		}
		datam.consensus = getConsensus(data, dim);
		this.data = upscaleAndSubtractConsensus(datam);
	}
	
	/**
	 * Calculates the consensus as mean of all samples stored column-wise.
	 * @param data
	 * @param dim
	 * @return
	 */
	private SimpleMatrix getConsensus(SimpleMatrix data, int dim){
		SimpleVector c = data.getCol(0);
		for(int i = 1; i < data.getCols(); i++){
			c.add(data.getCol(i));
		}
		c.divideBy(data.getCols());
		SimpleMatrix cons = new SimpleMatrix(data.getRows()/dim, dim);
		for(int i = 0; i < cons.getRows(); i++){
			for(int j = 0; j < cons.getCols(); j++){
				cons.setElementValue(i, j, c.getElement(i*dim + j));
			}
		}
		return cons;
	}
	
	/**
	 * Performs the principal component analysis on the data-set.
	 */
	public void run(){
		assert(data != null) : new Exception("Initialize data array fist.");
		
		System.out.println("Starting principal component analysis on " + numSamples + " data-sets.");
		
		DecompositionSVD svd = new DecompositionSVD(data);
		
		plot(svd.getSingularValues());
		
		this.numComponents = getPrincipalModesOfVariation(svd.getSingularValues());
		
		reduceDimensionality(svd.getSingularValues(), normalizeColumns(svd.getU()));
		//this.eigenVectors = normalizeColumns(svd.getU());
		//this.eigenValues = svd.getSingularValues();		
	}
	
	/**
	 * Performs the principal component analysis on the data-set.
	 * Alternative run with preset feature space dimensionality.
	 * @param dimensionality Number of principal components used to span the feature space.
	 */
	public void run(int dimensionality){
		assert(data != null) : new Exception("Initialize data array fist.");
		assert(dimensionality <= this.numSamples ) : new Exception("Feature space dimensionality cannot exceed number of samples.");
		
		System.out.println("Starting principal component analysis on " + numSamples + " data-sets.");
		
		DecompositionSVD svd = new DecompositionSVD(data);
		
		plot(svd.getSingularValues());
		
		this.numComponents = dimensionality;
		
		reduceDimensionality(svd.getSingularValues(), normalizeColumns(svd.getU()));
		//this.eigenVectors = normalizeColumns(svd.getU());
		//this.eigenValues = svd.getSingularValues();		
	}
	
	/**
	 * Allocates and sets the principal components and the corresponding variation values. Is used for dimensionality reduction after 
	 * the amount of principal components needed has been determined.
	 * @param v The variation values.
	 * @param pc The principal components.
	 */
	private void reduceDimensionality(double[] v, SimpleMatrix pc){
		this.eigenValues = new double[numComponents];
		this.eigenVectors = new SimpleMatrix(numPoints, numComponents);
		
		for(int i = 0; i < numComponents; i++){
			this.eigenVectors.setColValue(i, pc.getCol(i));
			this.eigenValues[i] = v[i];
		}
	}
	
	/** 
	 * Projects training shape num onto the principal components.
	 * @param num
	 * @return
	 */
	public double[] projectTrainingShape(int num){
		assert(this.eigenValues != null) : new Exception("Run analysis first.");
		// ASSUMES THAT CONSENSUS IS SUBTRACTED AND SCALE IS MULTIPLIED
		double[] weights = new double[numComponents];
		SimpleVector shape = data.getCol(num);
		for(int i = 0; i < numComponents; i++){
			SimpleVector comp = eigenVectors.getCol(i);
			double val = SimpleOperators.multiplyInnerProd(shape, comp);
			shape.subtract(comp.multipliedBy(val));
			weights[i] = val/eigenValues[i];
		}		
		//double error = shape.normL2()/shape.getLen();
		return weights;
	}
	
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
		
		if(PLOT_SINGULAR_VALUES){
			Plot plot = VisualizationUtil.createPlot(var, "Variation as function of principal component", "Principal Component", "Variation");
			plot.show();
		}
		
		return (i<ev.length)?i:ev.length;
	}
	
	/**
	 * Applies weighting to the columns in the score matrix and hence calculates weighted variation of the data corresponding 
	 * to a variation of the principal components. Weights are multiplied with the corresponding Eigenvalue.
	 * For the output points of dimensionality corresponding to the class member are assumed.
	 * Note that this the weights are formulated in terms of Variance not Standard-Deviation! Care has to be taken not to produce invalid shapes! 
	 * @param weights The weights for the different principal components.
	 * @return The variation as point-like matrix. 
	 */
	public SimpleMatrix applyWeight(float[] weights){
		assert(weights.length == this.eigenVectors.getCols()) : new Exception("Weights don't match the size of the score matrix.");
		SimpleVector col = new SimpleVector(numPoints);
		
		for(int i = 0; i < weights.length; i++){
			col.add(this.eigenVectors.getCol(i).multipliedBy(weights[i] * eigenValues[i]));
		}
		
		for(int i = 0; i < numVertices; i++){
			SimpleVector row = data.consensus.getRow(i);
			for(int j = 0; j < dimension; j++){
				col.addToElement(i * dimension + j, row.getElement(j));
			}
		}
		
		return toPointlikeMatrix(col);
	}
	
	/**
	 * Applies weighting to the columns in the score matrix and hence calculates weighted variation of the data corresponding 
	 * to a variation of the principal components. Weights are multiplied with the corresponding Eigenvalue.
	 * For the output points of dimensionality corresponding to the class member are assumed.
	 * Note that this the weights are formulated in terms of Variance not Standard-Deviation! Care has to be taken not to produce invalid shapes! 
	 * @param weights The weights for the different principal components.
	 * @return The variation as point-like matrix. 
	 */
	public SimpleMatrix applyWeight(double[] weights){
		assert(weights.length == this.eigenVectors.getCols()) : new Exception("Weights don't match the size of the score matrix.");
		SimpleVector col = new SimpleVector(numPoints);
		
		for(int i = 0; i < weights.length; i++){
			col.add(this.eigenVectors.getCol(i).multipliedBy(weights[i] * eigenValues[i]));
		}
		
		for(int i = 0; i < numVertices; i++){
			SimpleVector row = data.consensus.getRow(i);
			for(int j = 0; j < dimension; j++){
				col.addToElement(i * dimension + j, row.getElement(j));
			}
		}
		
		return toPointlikeMatrix(col);
	}
	
	/**
	 * Subtracts the consensus from the data-sets and re-scales the data-sets.
	 * @param mat The data-sets.
	 * @return The data-set after consensus subtraction.
	 */
	private DataMatrix upscaleAndSubtractConsensus(DataMatrix mat){
		
		for(int k = 0; k < numSamples; k++){
			float factor = mat.scaling.get(k);
			for(int i = 0; i < numVertices; i++){
				SimpleVector row = mat.consensus.getRow(i);
				for(int j = 0; j < dimension; j++){
					mat.multiplyElementBy(i * dimension + j, k, factor);
					mat.subtractFromElement(i * dimension + j, k, row.getElement(j));
				}
			}
		}
		return mat;
	}
	
	/**
	 * Scales the consensus data-set. The scaling used is the mean scaling of all data-sets in the sample.
	 * @param mat The data-sets.
	 * @return The data-set after consensus scaling.
	 */
	private DataMatrix scaleConsensus(DataMatrix mat){
		float scale = 0;
		
		for(int i = 0; i < numSamples; i++){
			scale += mat.scaling.get(i) / numSamples;
		}
		SimpleMatrix con = new SimpleMatrix(mat.consensus.getRows(), mat.consensus.getCols());
		for(int i = 0; i < numVertices; i++){
			for(int j = 0; j < dimension; j++){
				double val = mat.consensus.getElement(i,j) * scale;
				con.setElementValue(i, j, val);
			}
		}
		mat.consensus = con;
		return mat;
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
	 * Reshapes a vector to a matrix using the assumption, that the vector's entries are points of a certain dimension.
	 * @param vec The vector containing the data.
	 * @return	The reshaped data as matrix.
	 */
	private SimpleMatrix toPointlikeMatrix(SimpleVector vec){
		assert(vec.getLen() / dimension == numVertices) : new Exception("Dimensions don't match the input data.");
		
		SimpleMatrix mat = new SimpleMatrix(numVertices, dimension);
		
		for(int i = 0; i < numVertices; i++){
			for(int j = 0; j < dimension; j++){
				mat.setElementValue(i, j, vec.getElement(i * dimension + j));
			}
		}
		return mat;
	}
	
	/**
	 * Plots the data in the array over its array index.
	 * @param data The data to be plotted.
	 */
	private void plot(double[] data){
		if(PLOT_SINGULAR_VALUES){
			new ImageJ();
			Plot plot = VisualizationUtil.createPlot(data, "Singular values of data matrix", "Singular value", "Magnitude");
			plot.show();
		}
	}
	
	/**
	 * Returns the consensus object as one dimensional SimpleVector. Ordering is the same as in the data matrix:
	 * ( x_i y_i z_i x_(i+1) y_(i+1) z_(i+1) ... )
	 * @return The consensus object as single column vector.
	 */
	public SimpleVector getConsensus(){
		SimpleVector c = new SimpleVector(data.consensus.getCols() * data.consensus.getRows());
		for(int i = 0; i < data.consensus.getRows(); i++){
			for(int j = 0; j < data.consensus.getCols(); j++){
				c.setElementValue(i * data.consensus.getCols() + j, data.consensus.getElement(i, j));
			}
		}
		return c;
	}
	
	
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels;

import java.io.IOException;

import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.MeshUtil;
import edu.stanford.rsl.conrad.io.PcaIO;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.optimization.LSqMinNorm;

/**
 * This class contains members and methods to generate and store an active shape model.
 * Cootes, Timothy F., et al. "Active shape models-their training and application." Computer vision and image understanding 61.1 (1995): 38-59.
 * 
 * @author Mathias Unberath, Tobias Geimer
 * 
 * @version 2017-12-21 Now correctly uses standard deviation to scale the weights instead of variance in accordance to the {@link PCA} fix.
 *
 */
public class ActiveShapeModel {
	/**
	 * The Mesh object containing the final model. The model is the consensus plus a linear combination of the principal components, 
	 * weighted by both, their corresponding Eigenvalue and the weighting passed to the getModel method.
	 */
	public Mesh model;
	
	/**
	 * Dimension of the mesh's vertices.
	 */
	private int dimension;
	
	/**
	 * Number of points in the mesh assuming a dimension as stored in the class member.
	 */
	private int numPoints;
	
	/**
	 * The consensus shape.
	 */
	private SimpleVector consensus;
	
	/**
	 * Connectivity information for the consensus and hence all the active shape model.
	 */
	private SimpleMatrix connectivity;

	/**
	 * The number of components needed to  achieve the variability. Equals the dimension the principal components will be reduced to.
	 */
	public int numComponents;
	
	/**
	 * The principal components obtained by a Principal Component Analysis. Eigenvectors not needed to achieve the level of 
	 * variation asked for will be excluded, i.e. dimensionality reduction will be performed.
	 */
	private SimpleMatrix principalComponents;
	
	/**
	 * Root mean square error of a data-set fitting attempt.
	 */
	private double error;
	
	/**
	 * Variation coefficients corresponding to the principal components. Equal to the Eigenvalues obtained by Principal Component Analysis.
	 * They are used to determine the number of components needed to model a certain variance.
	 * Note that model weights are formulated in respect to standard deviation (i.e. square root of the eigenvalues) of the principal components. 
	 * E.g. most Active Shape Models limit their allowed values to [-3;3] times the standard deviation.
	 */
	private double[] variation;
	
	private SimpleMatrix fitRotation;
	private SimpleVector fitShift;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	/**
	 * Constructs the object and sets the necessary values for the construction of an Active shape Model. Performs dimensionality 
	 * reduction corresponding to the variability wanted.
	 * @param dimension The dimension of the vertex points in the model.
	 * @param variation The eigenvalues of the covariance matrix of the shapes as obtained by PCA.
	 * @param principalComponents The eigenvectors of the covariance matrix of the shapes as obtained by PCA.
	 * @param consensus The consensus object of all shapes used for GPA and PCA.
	 * @param connectivity The connectivity information for the consensus object.
	 * @param measure The variability measure for dimensionality reduction.
	 */
	public ActiveShapeModel(int dimension, double[] variation, SimpleMatrix principalComponents, SimpleVector consensus, SimpleMatrix connectivity, VarianceMeasure measure){
		this.dimension = dimension;
		this.numPoints = consensus.getLen() / dimension;
		this.connectivity = connectivity;
		this.consensus = consensus;
		
		this.numComponents = getNeededDimensionality(measure, variation);
		reduceDimensionality(variation, principalComponents);
	}
	
	/**
	 * Constructs the object and sets the necessary values for the construction of an Active shape Model. Performs dimensionality 
	 * reduction corresponding to the variability wanted.
	 * @param dimension The dimension of the vertex points in the model.
	 * @param variation The eigenvalues of the covariance matrix of the shapes as obtained by PCA.
	 * @param principalComponents The eigenvectors of the covariance matrix of the shapes as obtained by PCA.
	 * @param consensus The consensus object of all shapes used for GPA and PCA.
	 * @param connectivity The connectivity information for the consensus object.
	 * @param nC The number of principal components to be used.
	 */
	public ActiveShapeModel(int dimension, double[] variation, SimpleMatrix principalComponents, SimpleVector consensus, SimpleMatrix connectivity, int nC){
		this.dimension = dimension;
		this.numPoints = consensus.getLen() / dimension;
		this.connectivity = connectivity;
		this.consensus = consensus;
		
		assert(nC < variation.length) : new IllegalArgumentException("Number of principal components specified larger than principal components available with this model.");
		this.numComponents = nC;
		reduceDimensionality(variation, principalComponents);
	}
	
	/**
	 * Constructs the object and reads the principal components and consensus from the file specified in the argument. 
	 * Assumes a file as produced by the class PcaIO. Connectivity needs to be set manually.
	 * @param filename The file containing the principal components.
	 * @param measure The variability measure for dimensionality reduction.
	 */
	public ActiveShapeModel(String filename, VarianceMeasure measure){
		PcaIO reader = new PcaIO(filename);
		try {
			reader.readFile();
			this.consensus = reader.getConsensus();
			this.dimension = reader.getPointDimension();
			this.numPoints = consensus.getLen() / dimension;
			this.connectivity = reader.getConnectivity();
			
			this.numComponents = getNeededDimensionality(measure, reader.getEigenValues());
			reduceDimensionality(reader.getEigenValues(), reader.getEigenVectors());
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	/**
	 * Constructs the object and reads the principal components and consensus from the file specified in the argument. 
	 * Assumes a file as produced by the class PcaIO. Connectivity needs to be set manually.
	 * @param filename The file containing the principal components.
	 * @param nC The number of principal components to be used.
	 */
	public ActiveShapeModel(String filename, int nC){
		PcaIO reader = new PcaIO(filename);
		try {
			reader.readFile();
			this.consensus = reader.getConsensus();
			this.dimension = reader.getPointDimension();
			this.numPoints = consensus.getLen() / dimension;
			this.connectivity = reader.getConnectivity();
			
			assert(nC < variation.length) : new IllegalArgumentException("Number of principal components specified larger than principal components available with this model.");
			this.numComponents = nC;
			reduceDimensionality(reader.getEigenValues(), reader.getEigenVectors());
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	/**
	 * Constructs the object and reads the principal components and consensus from the file specified in the argument. 
	 * Assumes a file as produced by the class PcaIO.
	 * @param filename The file containing the principal components.
	 */
	public ActiveShapeModel(String filename){
		PcaIO reader = new PcaIO(filename);
		try {
			reader.readFile();
			this.consensus = reader.getConsensus();
			this.dimension = reader.getPointDimension();
			this.numPoints = consensus.getLen() / dimension;
			this.connectivity = reader.getConnectivity();
			
			this.principalComponents = reader.getEigenVectors();
			this.variation = reader.getEigenValues();
			this.numComponents = reader.getEigenValues().length;
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	/**
	 * Constructs an ASM from a PCA object.
	 * @param pca
	 */
	public ActiveShapeModel(PCA pca){
		this.consensus = pca.getConsensus();
		this.dimension = pca.dimension;
		this.numPoints = consensus.getLen() / dimension;
		this.connectivity = pca.connectivity;
		
		this.principalComponents = pca.eigenVectors;
		this.variation = pca.eigenValues;
		this.numComponents = pca.eigenValues.length;		
	}
	
	/**
	 * Sets the connectivity information for the model. Needed if the principal components are read from file.
	 * @param connectivity The connectivity information.
	 */
	public void setConnectivity(SimpleMatrix connectivity){
		this.connectivity = connectivity;
	}
	
	/**
	 * Constructs the Mesh object containing the final model. The model is constructed as linear combination of the principal 
	 * components, weighted by their corresponding Eigenvalue and the weight passed to the method, and added to the consensus.
	 * The weights need to fulfill several prerequisites to make sure, that only valid shapes are constructed.
	 * TODO add method to check for valid weights
	 * @param weights The weights for the principal components.
	 * @return The final model as Mesh object.
	 */
	public Mesh getModel(double[] weights){
		assert(weights.length == numComponents) : new Exception("Number of weights does not match number of principal components!");
		
		Mesh m = new Mesh();
		if(this.connectivity != null){
			m.setConnectivity(this.connectivity);
		}else{
			this.connectivity = new SimpleMatrix();
		}
		
		SimpleVector v = consensus.clone();
		for(int i = 0; i < numComponents; i++){
			// Model weights are formulated with respect to standard deviation, thus multiply with square root of the variance.
			v.add(this.principalComponents.getCol(i).multipliedBy(weights[i] * Math.sqrt(this.variation[i])));
		}		
		m.setPoints(toPointlikeMatrix(v));
		this.model = m;
		return m;
	}
	
	/**
	 * Constructs the Mesh object containing the final model. The model is constructed as linear combination of the principal 
	 * components, weighted by their corresponding Eigenvalue and the weight passed to the method, and added to the consensus.
	 * The weights need to fulfill several prerequisites to make sure, that only valid shapes are constructed.
	 * TODO add method to check for valid weights
	 * @param weights The weights for the principal components.
	 * @return The final model as Mesh object with the right pose.
	 */
	public Mesh getModelWithCorrectPose(double[] weights){
		assert(weights.length == numComponents) : new Exception("Number of weights does not match number of principal components!");
		
		Mesh m = new Mesh();
		if(this.connectivity != null){
			m.setConnectivity(this.connectivity);
		}else{
			this.connectivity = new SimpleMatrix();
		}
		
		SimpleVector v = consensus.clone();
		for(int i = 0; i < numComponents; i++){
			v.add(this.principalComponents.getCol(i).multipliedBy(weights[i] * Math.sqrt(this.variation[i])));
		}
		
		// new Shape aligned to consensus at Origin
		m.setPoints(toPointlikeMatrix(v));
		// new Shape with real rotation
		m = MeshUtil.rotate(m, fitRotation, false);
		// new Shape with real center
		m = MeshUtil.shift(m, fitShift);
		
		this.model = m;
		return m;
	}
	
	
	
	/**
	 * Allocates and sets the principal components and the corresponding variation values. Is used for dimensionality reduction after 
	 * the amount of principal components needed has been determined.
	 * @param v The variation values.
	 * @param pc The principal components.
	 */
	private void reduceDimensionality(double[] v, SimpleMatrix pc){
		this.variation = new double[numComponents];
		this.principalComponents = new SimpleMatrix(consensus.getLen(), numComponents);
		
		for(int i = 0; i < numComponents; i++){
			this.principalComponents.setColValue(i, pc.getCol(i));
			this.variation[i] = v[i];
		}
	}
	
	/**
	 * Calculates and returns the number of principal components needed to achieve model variability asked for.
	 * @param measure The measure determining the variability needed.
	 * @param variation The eigenvalues to be analyzed.
	 * @return The number of components.
	 */
	private int getNeededDimensionality(VarianceMeasure measure, double[] variation){
		return measure.evaluate(variation);
	}
	
	/**
	 * Reshapes a vector to a matrix using the assumption, that the vector's entries are points of a certain dimension.
	 * @param vec The vector containing the data.
	 * @return	The reshaped data as matrix.
	 */
	private SimpleMatrix toPointlikeMatrix(SimpleVector vec){
		assert(vec.getLen() / dimension == numPoints) : new Exception("Dimensions don't match the input data.");
		
		SimpleMatrix mat = new SimpleMatrix(numPoints, dimension);
		
		for(int i = 0; i < numPoints; i++){
			for(int j = 0; j < dimension; j++){
				mat.setElementValue(i, j, vec.getElement(i * dimension + j));
			}
		}
		return mat;
	}
	
	/**
	 * Calculates the weighting of principal components needed in order to model the input shape.
	 * The system of equations is solved using an optimization scheme as described in the solver.
	 * The shape will be centered and aligned with the consensus during this procedure.
	 * The fitting error obtained by this method assumes a point dimension of 1 and should hence not be used in other cases!
	 * @param shapeMat The shape to be expressed in terms of the ActiveShapeModel.
	 * @return The weighting needed to express the shape.
	 */
	public double[] fitModelToShape(SimpleMatrix shapeMat){
		assert(shapeMat.getCols() == this.dimension) : new Exception("Input shape vertex dimension does not match Active Shape Models' vertex dimension.");
		assert(shapeMat.getRows() == this.numPoints) : new Exception("Input shape number of vertices does not match Active Shape Models' number of vertices. Can be solved using ICP.");
		
		SimpleMatrix centeredAndAlignedMat = alignWithConsensus(centerShape(shapeMat));
		
		SimpleVector shape = toSimpleVector(centeredAndAlignedMat);
		shape.subtract(consensus);
		
		LSqMinNorm solver = new LSqMinNorm(principalComponents, shape);
		
		// the solver is very general so we have to divide by the variance of the principal component first in order to use it for model generation
		double[] weights = solver.getSolution();
		for(int i = 0; i < numComponents; i++){
			weights[i] /= Math.sqrt(this.variation[i]);
		}
		
		this.error = solver.getRmsError();
		
		return weights;
	}
	
	/**
	 * Calculates the weighting of principal components needed in order to model the input shape.
	 * The weights are calculated using projections on each principal component.
	 * The shape will be centered and aligned with the consensus during this procedure.
	 * @param shapeMat The shape to be expressed in terms of the ActiveShapeModel.
	 * @return The weighting needed to express the shape.
	 */
	public double[] projectShape(SimpleMatrix shapeMat){
		assert(shapeMat.getCols() == this.dimension) : new Exception("Input shape vertex dimension does not match Active Shape Models' vertex dimension.");
		assert(shapeMat.getRows() == this.numPoints) : new Exception("Input shape number of vertices does not match Active Shape Models' number of vertices. Can be solved using ICP.");
		
		SimpleMatrix centeredAndAlignedMat = alignWithConsensus(centerShape(shapeMat));
		
		SimpleVector shape = toSimpleVector(centeredAndAlignedMat);
		shape.subtract(consensus);

		double[] weights = new double[numComponents];
		for(int i = 0; i < numComponents; i++){
			SimpleVector comp = principalComponents.getCol(i);
			double val = SimpleOperators.multiplyInnerProd(shape, comp);
			shape.subtract(comp.multipliedBy(val));
			weights[i] = val/Math.sqrt(this.variation[i]);
		}
		
		double val = 0;
		for(int i = 0; i < numPoints; i++){
			SimpleVector diff = new SimpleVector(dimension);
			for(int j = 0; j < dimension; j++){
				diff.setElementValue(j, shape.getElement(i*dimension +j));
			}
			val += diff.normL2();
		}
		
		this.error = val/numPoints;
		
		return weights;
	}
	
	/**
	 * Calculates the weighting of principal components needed in order to model the input shape.
	 * The weights are calculated using projections on each principal component.
	 * The shape will be centered and aligned with the consensus during this procedure.
	 * @param shapeMat The shape to be expressed in terms of the ActiveShapeModel.
	 * @return The resulting shape
	 */
	public Mesh getProjectedShape(SimpleMatrix shapeMat){
		assert(shapeMat.getCols() == this.dimension) : new Exception("Input shape vertex dimension does not match Active Shape Models' vertex dimension.");
		assert(shapeMat.getRows() == this.numPoints) : new Exception("Input shape number of vertices does not match Active Shape Models' number of vertices. Can be solved using ICP.");
		
		SimpleMatrix centeredAndAlignedMat = alignWithConsensus(centerShape(shapeMat));
		
		SimpleVector shape = toSimpleVector(centeredAndAlignedMat);
		shape.subtract(consensus);

		double[] weights = new double[numComponents];
		for(int i = 0; i < numComponents; i++){
			SimpleVector comp = principalComponents.getCol(i);
			double val = SimpleOperators.multiplyInnerProd(shape, comp);
			shape.subtract(comp.multipliedBy(val));
			weights[i] = val/Math.sqrt(this.variation[i]);
		}
		
		double val = 0;
		for(int i = 0; i < numPoints; i++){
			SimpleVector diff = new SimpleVector(dimension);
			for(int j = 0; j < dimension; j++){
				diff.setElementValue(j, shape.getElement(i*dimension +j));
			}
			val += diff.normL2();
		}
		
		this.error = val/numPoints;
		
		Mesh fit = getModel(weights);
		SimpleMatrix rot = fitRotation;
		SimpleVector trans = fitShift;
		SimpleMatrix pts = fit.getPoints();
		for(int i = 0; i < fit.numPoints; i++){
			SimpleVector pt = SimpleOperators.multiply(rot, pts.getRow(i));
			pt.add(trans);
			pts.setRowValue(i, pt);
		}
		fit.setPoints(pts);
		return fit;
	}
	
	/**
	 * Calculates the weighting of principal components needed in order to model the input shape.
	 * The system of equations is solved using an optimization scheme as described in the solver.
	 * The shape will be centered and aligned with the consensus during this procedure.
	 * The fitting error obtained by this method assumes a point dimension of 1 and should hence not be used in other cases!
	 * @param shapeMat The shape to be expressed in terms of the ActiveShapeModel.
	 * @return fitted shape, centroid shifted and rotated to match the input
	 */
	public Mesh fitToShape(SimpleMatrix shapeMat){
		assert(shapeMat.getCols() == this.dimension) : new Exception("Input shape vertex dimension does not match Active Shape Models' vertex dimension.");
		assert(shapeMat.getRows() == this.numPoints) : new Exception("Input shape number of vertices does not match Active Shape Models' number of vertices. Can be solved using ICP.");
		
		SimpleMatrix centeredAndAlignedMat = alignWithConsensus(centerShape(shapeMat));
		
		SimpleVector shape = toSimpleVector(centeredAndAlignedMat);
		shape.subtract(consensus);
		
		LSqMinNorm solver = new LSqMinNorm(principalComponents, shape);
		
		// the solver is very general so we have to divide by the variance of the principal component first in order to use it for model generation
		double[] weights = solver.getSolution();
		for(int i = 0; i < numComponents; i++){
			weights[i] /= Math.sqrt(this.variation[i]);
		}		
		this.error = solver.getRmsError();
		
		Mesh fit = getModel(weights);
		SimpleMatrix rot = fitRotation;
		SimpleVector trans = fitShift;
		SimpleMatrix pts = fit.getPoints();
		for(int i = 0; i < fit.numPoints; i++){
			SimpleVector pt = SimpleOperators.multiply(rot, pts.getRow(i));
			pt.add(trans);
			pts.setRowValue(i, pt);
		}
		fit.setPoints(pts);
		return fit;
	}
	
	/**
	 * Calculates the weighting of principal components needed in order to model the input shape.
	 * The system of equations is solved using an optimization scheme as described in the solver.
	 * @param shapeMesh The shape to be expressed in terms of the ActiveShapeModel.
	 * @return The weighting needed to express the shape.
	 */
	public double[] fitModelToShape(Mesh shapeMesh){
		assert(shapeMesh.getPoints().getCols() == this.dimension) : new Exception("Input shape vertex dimension does not match Active Shape Models' vertex dimension.");
		assert(shapeMesh.getPoints().getRows() == this.numPoints) : new Exception("Input shape number of vertices does not match Active Shape Models' number of vertices. Can be solved using ICP.");
		
		Mesh aligned = MeshUtil.centerToOrigin(shapeMesh);
		aligned = MeshUtil.rotate(toPointlikeMatrix(consensus), aligned, true);
		
		SimpleVector shape = toSimpleVector(aligned.getPoints());
		shape.subtract(consensus);
		
		LSqMinNorm solver = new LSqMinNorm(principalComponents, shape);
		
		// the solver is very general so we have to divide by the variance of the principal component first in order to use it for model generation
		double[] weights = solver.getSolution();
		for(int i = 0; i < numComponents; i++){
			weights[i] /= Math.sqrt(this.variation[i]);
		}
		
		this.error = solver.getRmsError();
		
		return weights;
	}
	
	/**
	 * Getter for the RMS error after model fitting.
	 * @return The error.
	 */
	public double getFittingError(){
		return this.error;
	}
	
	/**
	 * Transforms a SimpleMatrix into a SimpleVector by appending each consecutive row to the former.
	 * @param m The SimpleMatrix.
	 * @return The SimpleMatrix as SimpleVector.
	 */
	private SimpleVector toSimpleVector(SimpleMatrix m){
		SimpleVector v = new SimpleVector(m.getRows() * m.getCols());
		for(int i = 0; i < m.getRows(); i++){
			for(int j = 0; j < m.getCols(); j++){
				v.setElementValue(i * m.getCols() + j, m.getElement(i, j));
			}
		}
		return v;
	}
	
	/**
	 * Aligns a shape matrix to the consensus object of the active shape model for fitting purposes.
	 * @param m2
	 * @return aligned shape
	 */
	private SimpleMatrix alignWithConsensus(SimpleMatrix m2){
		SimpleMatrix m1 = new SimpleMatrix(m2.getRows(), m2.getCols());
		for(int i = 0; i < m2.getRows(); i++){
			for(int j = 0; j < m2.getCols(); j++){
				m1.setElementValue(i, j, consensus.getElement(i*dimension + j));
			}
		}
		// create matrix containing information about both point-clouds m1^T * m2
		SimpleMatrix m1Tm2 = SimpleOperators.multiplyMatrixProd(m1.transposed(), m2);
		// perform SVD such that:
		// m1^T * m2 = U sigma V^T
		DecompositionSVD svd = new DecompositionSVD(m1Tm2, true, true, true);
		// exchange sigma with new matrix s having only +/- 1 as singular values
		// this allows only for rotations but no scaling, e.g. sheer
		// signum is the same as in sigma, hence reflections are still taken into account
		int nColsS = svd.getS().getCols();
		SimpleMatrix s = new SimpleMatrix(nColsS,nColsS);
		for(int i = 0; i < nColsS; i++){
			s.setElementValue(i, i, Math.signum(svd.getSingularValues()[i]));			
		}
		// calculate rotation matrix such that:
		// H = V s U^T
		SimpleMatrix h = SimpleOperators.multiplyMatrixProd(svd.getV(), SimpleOperators.multiplyMatrixProd(s, svd.getU().transposed()));
		this.fitRotation = h;
		return SimpleOperators.multiplyMatrixProd(m2, h);
	}
	
	/**
	 * Centers the shape passed to the method, assuming that vertices are stored row-wise in the shape matrix.
	 * @param shapeMat
	 * @return centered shape
	 */
	private SimpleMatrix centerShape(SimpleMatrix shapeMat){
		SimpleVector mean = new SimpleVector(shapeMat.getCols());
		for(int i = 0; i < shapeMat.getRows(); i++){
			mean.add(shapeMat.getRow(i));
		}
		mean.divideBy(shapeMat.getRows());
		SimpleMatrix centered = new SimpleMatrix(shapeMat.getRows(), shapeMat.getCols());
		for(int i = 0; i < shapeMat.getRows(); i++){
			SimpleVector row = shapeMat.getRow(i);
			row.subtract(mean);
			centered.setRowValue(i, row);
		}
		this.fitShift = mean;
		return centered;
	}
	
	
	public double[] getEigenvalues(){
		return this.variation;
	}
	
	public double[] getStandardDeviation() {
		double[] sd = new double[this.variation.length];
		for( int i = 0; i < this.variation.length; i++ ) {
			sd[i] = Math.sqrt(this.variation[i]);
		}
		return sd;
	}
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

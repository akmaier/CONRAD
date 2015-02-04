/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


/**
 * This class implements the Generalized Procrustes Analysis using the method proposed by J. C. Gower.
 * Assumes that perfect point correspondence between the point-clouds is established.
 * @see Generalized Procrustes Analysis, J.C. Gower (1975), Psychometrika Vol. 40 pp. 33-51
 * @author Mathias Unberath
 *
 */
public class GPA {

	/**
	 * Dimension of the point-clouds' vertices.
	 */
	public int dimension = -1;
	
	/**
	 * Value at which the Generalized Procrustes Analysis iteration scheme shall be stopped 
	 * because convergence is supposed to be reached. 
	 * This value is expressed in % of initial error.
	 */
	public float convergence = 1e-3f;
	
	/**
	 * Number of points in one point-cloud.
	 */
	public int numPoints = -1;
	
	/**
	 * Array to store the scaling factor for the single point-clouds stored in the point ArrayList.
	 * The factor is used to re-scale the meshes during Generalized Procrustes Alignment.
	 */
	public ArrayList<Float> scaling;
	
	/**
	 * Scaling used during Generalized Procrustes Analysis to maintain the proper scaling of the point-clouds.
	 */
	public ArrayList<Float> rho;
	
	/**
	 * Array to store the centers of mass of the single point-clouds stored in the point ArrayList.
	 * The centers are used to shift the point-clouds' mean value to the origin during 
	 * Generalized Procrustes Alignment.
	 */
	public ArrayList<PointND> centers;
	
	/**
	 * ArrayList to store the rotation matrices for each point-cloud.
	 */
	public ArrayList<SimpleMatrix> rotations;
	
	/**
	 * An ArrayList containing the point-clouds' vertices.
	 */
	public ArrayList<SimpleMatrix> pointList;
	
	/**
	 * The consensus pint-cloud calculated during Generalized Procrustes Analysis.
	 */
	public SimpleMatrix consensus;
	
	/**
	 * The connectivity information for the case, where the point-clouds are meshes and hence 
	 * connectivity information exists. Is not used within this class but can be passed on.
	 * One single copy of the connectivity is enough as perfect point correspondence between 
	 * the point-clouds is assumed. 
	 */
	public SimpleMatrix connectivity;
	
	/**
	 * NUmber of point-clouds to be analyzed during this GPA run.
	 * Note: using the method addElement will increase this number, while using addElementAtIndex will assume 
	 * that enough elements are initialized.
	 */
	public int numPc = 0;
	
	/**
	 * Debug flag for console output.
	 */
	public boolean DEBUG = true;
	
	public boolean PRINT = false;
	//==========================================================================================
	// METHODS
	//==========================================================================================\
	
	/**
	 * Creates the object and initializes the point-cloud list.
	 * @param numPointClouds Number of point-clouds used for this GPA run.
	 */
	public GPA(int numPointclouds){
		assert(numPointclouds > 0) : new IllegalArgumentException("Number of point-clouds must be bigger than 0.");
		this.numPc = numPointclouds;
		this.pointList = new ArrayList<SimpleMatrix>(numPc);
	}
	
	/** 
	 * Start the Generalized Procrustes Alignment on the point-cloud data.
	 */
	public void runGPA(){
		System.out.println("Starting Generalized Procrustes Analysis on " + numPc + " data-sets.");
		
		this.rotations = new ArrayList<SimpleMatrix>(numPc);
		
		shiftMeanToOriginAndScale();
		
		// first consensus object is the first element
		this.consensus = pointList.get(0);
		// initialize the intra-procedure scalings
		initializeRho();
		
		int nIter = 1;
		if(DEBUG){
			System.out.println("Iteration: " + nIter);
			System.out.println("Calculating first consensus object.");
		}
		nIter++;
		// first iteration is slightly different in residual and rho-scaling
		for(int i = 0; i < numPc; i++){
			getRotationMatrixAndRotate(i);
		}
		updateConsensus();
		
		double residual = getInitialResidual();
		double initialRes = residual;
		
		if(DEBUG){
			System.out.println("Initial residual: " + residual);
		}
		
		double oldRes = 0;
		double change = 1;
		
		while(change > initialRes/100*convergence){	// this should be changed according to convergence criteria
			if(PRINT){
				// do something
			}
			
			oldRes = residual;
			if(DEBUG){
				System.out.println("Iteration: " + nIter);
			}
			// first step: rotate every cloud, then update consensus and residual
			for(int i = 0; i < numPc; i++){
				getRotationMatrixAndRotate(i);
			}
			SimpleMatrix oldConsensus = consensus;
			updateConsensus();
			residual -= getResidual(oldConsensus);
			
			if(DEBUG){
				System.out.println("Residual at rotation " + nIter + " is: " + residual);
			}
			// second step: re-scale every cloud, then update consensus and residual
			updateRho();
			rescalePointClouds();
			oldConsensus = consensus;
			updateConsensus();
			residual -= getResidual(oldConsensus);
			
			if(DEBUG){
				System.out.println("Residual at rescaling " + nIter + " is: " + residual);
			}			
			// check if we have converged:
			change = oldRes - residual;
			if(DEBUG){
				System.out.println("Residual change at iteration " + nIter + " is: " + change);
			}
			nIter++;
		}
		
	}
	
	/**
	 * Initializes the object and initializes the point-cloud list.
	 * @param numPointClouds Number of point-clouds used for this GPA run.
	 */
	public void init(int numPointclouds){
		assert(numPointclouds > 0) : new IllegalArgumentException("Number of point-clouds must be bigger than 0.");
		this.numPc = numPointclouds;
		this.pointList = new ArrayList<SimpleMatrix>(numPc);
	}
	
	/**
	 * Adds a point-cloud to the list and increments the number of point-clouds.
	 * @param pointclod Point-cloud to add.
	 */
	public void addElement(SimpleMatrix pointcloud){
		if(numPoints == -1){
			this.numPoints = pointcloud.getRows();
			this.dimension = pointcloud.getCols();
		}else{
			assert(numPoints == pointcloud.getRows()) : new IllegalArgumentException("Number of points in point-cloud does not match.");
			assert(dimension == pointcloud.getCols()) : new IllegalArgumentException("Point dimension in point-cloud does not match.");
		}
		
		this.pointList.add(pointcloud);
		this.numPc++;
	}
	
	/**
	 * Adds a point-cloud to the list at a certain index.
	 * @param idx Position where point-cloud will be added.
	 * @param pointclod Point-cloud to add.
	 */
	public void addElement(int idx, SimpleMatrix pointcloud){
		if(numPoints == -1){
			this.numPoints = pointcloud.getRows();
			this.dimension = pointcloud.getCols();
		}else{
			assert(numPoints == pointcloud.getRows()) : new IllegalArgumentException("Number of points in point-cloud does not match.");
			assert(dimension == pointcloud.getCols()) : new IllegalArgumentException("Point dimension in point-cloud does not match.");
		}
		assert(idx < numPc) : new IllegalArgumentException("Index out of bounds. Use addElement() instead.");
		
		this.pointList.add(idx, pointcloud);
	}
	
	/**
	 * Adds the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	private void addCenterOfMassAtIndex(int colIdx, PointND centOfMass){
		assert(colIdx < numPc) : new IllegalArgumentException("Index out of bounds. Initialize list first.");
		this.centers.add(colIdx, centOfMass);
	}
	
	/**
	 * Adds the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	private void addScalingAtIndex(int colIdx, float scaling){
		assert(colIdx < this.numPc) : new IllegalArgumentException("Index out of bounds. Initialize list first.");
		this.scaling.add(colIdx, scaling);
	}
	
	/**
	 * Sets the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	public void setCenterOfMassAtIndex(int colIdx, PointND centOfMass){
		assert(colIdx < this.numPc) : new IllegalArgumentException("Index out of bounds. Add point-cloud first.");
		this.centers.set(colIdx, centOfMass);
	}
	
	/**
	 * Sets the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	public void setScalingAtIndex(int colIdx, float scaling){
		assert(colIdx < this.numPc) : new IllegalArgumentException("Index out of bounds. Add point-cloud first.");
		this.scaling.set(colIdx, scaling);
	}
	
	/**
	 * This method calculates the centroid and scaling-factor for all point-clouds stored in the list. The centroid and scaling-factor 
	 * are stored in the corresponding class members. The point-clouds centroid is then shifted to the origin and scaling is applied.
	 */
	private void shiftMeanToOriginAndScale(){
		assert(numPoints > 0 || dimension > 0) : new Exception("No data for GPA found. Did you add the datasets?");
		checkPointList();
		
		this.centers = new ArrayList<PointND>(numPc);
		this.scaling = new ArrayList<Float>(numPc);
		
		double val;
		
		
		System.out.println("Calculating centroid and scaling for each point-cloud.");
		
		for(int k = 0; k < numPc; k++){
			SimpleMatrix cloud = pointList.get(k);
			
			SimpleVector mean = new SimpleVector(dimension);
			// calculate mean value
			float[] mAcc = new float[3];
			for(int i = 0; i < numPoints; i++){
				for(int j = 0; j < dimension; j++){
					val = cloud.getElement(i, j);
					mAcc[j] += val;
				}
			}
			
			for(int j = 0; j < dimension; j++){
				mean.setElementValue(j, mAcc[j]/numPoints);
			}
			addCenterOfMassAtIndex(k, new PointND(mean));
			
			// calculate scale value using mean value
			float sAcc = 0;
			for(int i = 0; i < numPoints; i++){
				for(int j = 0; j < dimension; j++){
					val = cloud.getElement(i, j);
					sAcc += Math.pow((val - mean.getElement(j)), 2);
				}
			}
			float scaling = (float)Math.sqrt(sAcc);
			addScalingAtIndex(k, scaling);
			
			// updated point-cloud
			for(int i = 0; i < numPoints; i++){
				for(int j = 0; j < dimension; j++){
					val = cloud.getElement(i, j);
					cloud.setElementValue(i, j, (val - mean.getElement(j)) / scaling);
				}
			}
			pointList.set(k,cloud);
		}		
	}
	
	/**
	 * Checks if all initialized list entries have been filled with point-clouds.
	 */
	private void checkPointList(){
		assert (pointList.size() == numPc) : new Exception("Not all initialized point-clouds set.");
	}
	
	/**
	 * Computes the rotation matrix for the RMS-norm-minimal rotational registration of consensus and point-cloud at index idx.
	 * Then calls the method to store the rotation  matrix in the class member list and rotate the point-cloud.
	 * Uses a singular value decomposition.
	 * @param idx	The index of the point-cloud.
	 */
	private void getRotationMatrixAndRotate(int idx){
		SimpleMatrix m1 = consensus;
		SimpleMatrix m2 = pointList.get(idx);

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
		
		rotatePointCloudAtIndex(idx, h);
	}
	
	/**
	 * This method rotates the point-cloud at a certain index. The rotation has to be pre-computed and expressed as a matrix. 
	 * The rotation matrix will be stored in the corresponding class member in order to be able to restore the original point-cloud.
	 * @param idx	Point-cloud index in list.
	 * @param rotation	Rotation matrix to be applied.
	 */
	private void rotatePointCloudAtIndex(int idx, SimpleMatrix rotation){
		// decide whether it's the first rotation or a follow up
		// first entries have to be added, follow ups need to be set
		if(rotations.size() != numPc){
			this.rotations.add(rotation);
		}else{
			this.rotations.set(idx, rotation);
		}
		// rotate the point-cloud
		this.pointList.set(idx, SimpleOperators.multiplyMatrixProd(pointList.get(idx), rotation));
	}
	
	/**
	 * Calculates the initial residual of the conensus point-cloud in the sense of J. C. Gower.
	 * @return
	 */
	private float getInitialResidual(){
		SimpleMatrix  res = SimpleOperators.multiplyMatrixProd(consensus.transposed(), consensus);
		
		float residual = numPc * (1 - trace(res));
		return residual;
	}
	
	/** 
	 * Calculates the residual update for all iterations except the initial one.
	 * @param oldConsensus The old consensus point-cloud.
	 * @return The change in residual.
	 */
	private float getResidual(SimpleMatrix oldConsensus){
		SimpleMatrix M1 = SimpleOperators.multiplyMatrixProd(consensus.transposed(), consensus);
		SimpleMatrix M2 = SimpleOperators.multiplyMatrixProd(oldConsensus.transposed(), oldConsensus);		
		float residual = numPc * (trace(M1) - trace(M2));
		return residual;
	}
	
	/**
	 * Updates the consensus point-cloud. The new consensus is the mean of all point-clouds stored in the list.
	 */
	private void updateConsensus(){
		SimpleMatrix cons = new SimpleMatrix(numPoints,dimension); 
		for(int k = 0; k < numPc; k++){
			SimpleMatrix pc = pointList.get(k);
			 for(int i = 0; i < numPoints; i++){
				 for(int j = 0; j < dimension; j++){
					 cons.addToElement(i, j, pc.getElement(i, j)/numPc);
				 }
			 }
		 }
		this.consensus = cons;
	}
	
	/**
	 * Rescales the point-clouds with the new rho.
	 */
	private void rescalePointClouds(){ 
		for(int k = 0; k < numPc; k++){
			SimpleMatrix pc = pointList.get(k);
			float rho = this.rho.get(k);
			 for(int i = 0; i < numPoints; i++){
				 for(int j = 0; j < dimension; j++){
					 
					 pc.multiplyElementBy(i, j, rho);
				 }
			 }
			 this.pointList.set(k, pc);
		 }
	}
	
	/**
	 * Initializes the GPA scaling rho with all ones.
	 */
	private void initializeRho(){
		this.rho = new ArrayList<Float>(numPc);
		for(int i = 0; i < numPc; i++){
			this.rho.add(1f);
		}
	}
	
	/**
	 * Calculates the new scaling rho update using the new consensus object. The updates rho is given as 
	 * ratio of the new rho divided by the old rho value to compensate for the old scaling already being 
	 * applied to the point-cloud. 
	 */
	private void updateRho(){
		
		float traceConsensus = trace(SimpleOperators.multiplyMatrixProd(consensus.transposed(), consensus));
		
		for(int k = 0; k < numPc; k++){
			SimpleMatrix cloud = pointList.get(k);
			float traceCloud = trace(SimpleOperators.multiplyMatrixProd(cloud.transposed(), cloud));
			float traceCloudCons = trace(SimpleOperators.multiplyMatrixProd(cloud.transposed(), consensus));
			float rhoUpdate = (float)Math.sqrt(traceCloudCons / (traceConsensus * traceCloud));
			this.rho.set(k, rhoUpdate);
			//this.scaling.set(k, this.scaling.get(k)/rhoUpdate);
			//this.rho.set(k, 1f);
			
		}
	}
	
	/**
	 * Calculates the trace of a square matrix.
	 * @param m	The square Matrix.
	 * @return	The trace of the matrix.
	 */
	private float trace(SimpleMatrix m){
		assert(m.isSquare()) : new IllegalArgumentException("Trace not defined for non-square matrices.");
		float tr = 0;
		for(int i = 0; i < m.getRows(); i++){
			tr += m.getElement(i, i);
		}
		return tr;
	}
	
	/**
	 * Calculates a scaled and centroid-shifted version of the consensus object.
	 * The scaling is the mean scaling of the input point-clouds.
	 * The centroid is the mean of the centroids.
	 * @return A SimpleMatrix containing the consensus point-cloud.
	 */
	public SimpleMatrix getScaledAndShiftedConsensus(){
		SimpleMatrix m = new SimpleMatrix(numPoints,dimension);
		
		// calculate mean-scale and mean of centroids
		float upScale = 0;
		SimpleVector meanCentroid = new SimpleVector(dimension);
		for(int i = 0; i < numPc; i++){
			upScale += scaling.get(i) / numPc;
			for(int j = 0; j < dimension; j++){
				meanCentroid.addToElement(j, centers.get(i).get(j) / numPc);
			}
		}
		// apply scaling
		m = consensus.multipliedBy(upScale);
		// shift by mean centroid
		for(int j = 0; j < dimension; j++){
			float val = (float) meanCentroid.getElement(j);
			for(int i = 0; i < numPoints; i++){
				m.addToElement(i, j, val);
			}
		}		
		
		return m;
	}
	
	/**
	 * Calculats the up-scaled and centroid shifted version of the point-cloud at index <idx> after 
	 * Generalized Procrustes Alignment.
	 * @param idx Index of point-cloud to be processed.
	 * @return The up-scaled and centroid shifted point-cloud.
	 */
	public SimpleMatrix getScaledAndShiftedPointCloud(int idx){
		SimpleMatrix m = new SimpleMatrix(numPoints,dimension);
		
		// apply scaling
		m = pointList.get(idx).multipliedBy(scaling.get(idx));
		// shift by mean centroid
		for(int j = 0; j < dimension; j++){
			float val = (float) centers.get(idx).get(j);
			for(int i = 0; i < numPoints; i++){
				m.addToElement(i, j, val);
			}
		}			
		
		return m;
	}
	
	/**
	 * Sets the connectivity information in case it exists for the point-clouds.
	 * @param con The connectivity information.
	 */
	public void setConnectivity(SimpleMatrix con){
		this.connectivity = con;
	}
	
	/**
	 * Calculates and returns the mean center of mass of all point-clouds.
	 * @return The mean center of mass.
	 */
	public SimpleVector getMeanCenter(){
		SimpleVector mC = new SimpleVector(dimension);
		for(int i = 0; i < centers.size(); i++){
			mC.add(centers.get(i).getAbstractVector());
		}
		return mC.dividedBy(centers.size());
	}
	
}

/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

































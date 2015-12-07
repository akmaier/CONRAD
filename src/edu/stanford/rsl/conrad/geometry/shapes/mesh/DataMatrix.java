/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.mesh;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.GPA;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This class stores meshes as column vectors in a SimpleMatrix structures. Centers of mass and scalings can be stored in ArrayLists as well, 
 * so that original positions and sizes can be restored after e.g. a zero-mean-shift.
 * @author Mathias Unberath
 *
 */
public class DataMatrix extends SimpleMatrix{

	private static final long serialVersionUID = 1L;

	/**
	 * Dimension of the data-set elements, needed due to single column storage.
	 */
	public int dimension = 0;
	
	/**
	 * Array to store the scaling factor for the single meshes stored in the SimpleMatrix.
	 * The factor can be used to scale the meshes to a common size, needed in e.g. Generalized Procrustes Alignment.
	 */
	public ArrayList<Float> scaling;
	
	/**
	 * Array to store the centers of mass of the single meshes stored in the SimpleMatrix.
	 * The centers can be used to restore the meshes' position after shifting the mean value to the origin, 
	 * needed in e.g. Generalized Procrustes Alignment.
	 */
	public ArrayList<PointND> centers;
	
	/**
	 * The consensus object of the data-sets after GPA.
	 */
	public SimpleMatrix consensus;
	
	/**
	 * Flag to show whether or not connectivity information has been provided.
	 */
	public boolean HAS_CONNECTIVITY = false;
	
	/**
	 * List to store the connectivity information. One list is enough as we assume that the meshes have the same 
	 * amount of vertices and point correspondence to be established beforehand.
	 */
	public SimpleMatrix connectivity;
	
	/**
	 * Flag to check whether or not a GPA object has been used for construction. Needed for the initialization of 
	 * the ArrayLists.
	 */
	private boolean USE_GPA = false;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	/**
	 * Convenience constructor. As this class is intended to be used for principal component analysis after a 
	 * generalized procrustes analysis, the object can be constructed directly from a GPA object. 
	 * @param gpa
	 */
	public DataMatrix(GPA gpa){
		USE_GPA = true;
		
		this.connectivity = gpa.connectivity;
		if(gpa.connectivity !=  null){
			this.HAS_CONNECTIVITY = true;
		}
		this.centers = gpa.centers;
		this.scaling = gpa.scaling;
		this.consensus = gpa.consensus;
		
		this.dimension = gpa.dimension;
		
		init(gpa.numPoints * dimension, gpa.numPc);
		
		for(int i = 0; i < cols; i++){
			setSimpleMatrixAtIndex(gpa.pointList.get(i), i);
		}
	}
	
	public DataMatrix() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * Sets the number of rows via the number of mesh vertices and their dimension.
	 * Therefore the number of rows is #dimension times the number of mesh vertices. 
	 * Sets the number of columns via the number of meshes used for the model. The meshes are stored column-wise
	 * in the matrix. Therefore the number of meshes is equal to the number of columns.
	 * @param numPoints The number of mesh vertices.
	 * @param dim Dimension of vertices.
	 * @param numMesh The number of meshes used for the analysis.
	 */
	public void setDimensions(int numPoints, int dim, int numMesh){
		this.dimension = dim;
		int cols = numMesh;
		int rows = dimension * numPoints;
		init(rows,cols);
	}
	
	/**
	 * Sets the number of rows via the number of mesh vertices. 3 dimensional points are assumed.
	 * Therefore the number of rows is 3 times the number of mesh vertices. 
	 * Sets the number of columns via the number of meshes used for the model. The meshes are stored column-wise
	 * in the matrix. Therefore the number of meshes is equal to the number of columns.
	 * @param numPoints The number of mesh vertices.
	 * @param numMesh The number of meshes used for the analysis.
	 */
	public void setDimensions(int numPoints, int numMesh){
		int cols = numMesh;
		
		if(dimension == 0){
			System.out.println("Dimension of elements has not been set. Assuming 3D.");
			this.dimension = 3;
		}
		
		int rows = dimension * numPoints; // assumes 3 dimensional points in mesh!
		init(rows,cols);
	}
	
	/**
	 * Adds the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	public void addCenterOfMassAtIndex(int colIdx, PointND centOfMass){
		assert(colIdx < cols) : new IllegalArgumentException("Column index for this mesh-matrix is out of bounds.");
		this.centers.add(colIdx, centOfMass);
	}
	
	/**
	 * Adds the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	public void addScalingAtIndex(int colIdx, float scaling){
		assert(colIdx < this.cols) : new IllegalArgumentException("Column index for this mesh-matrix is out of bounds.");
		this.scaling.add(colIdx, scaling);
	}
	
	/**
	 * Sets the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	public void setCenterOfMassAtIndex(int colIdx, PointND centOfMass){
		assert(colIdx < this.cols) : new IllegalArgumentException("Column index for this mesh-matrix is out of bounds.");
		this.centers.set(colIdx, centOfMass);
	}
	
	/**
	 * Sets the center of mass of a mesh object at a certain index position. Needed to restore original mesh positions after 
	 * a shift to zero-mean.
	 * @param colIdx	List-index to be written.
	 * @param centOfMass	Value to be written at list-index.
	 */
	public void setScalingAtIndex(int colIdx, float scaling){
		assert(colIdx < this.cols) : new IllegalArgumentException("Column index for this mesh-matrix is out of bounds.");
		this.scaling.set(colIdx, scaling);
	}
	
	/**
	 * Writes the vertex coordinates of a mesh object into the <colIdx> column of a SimpleMatrix structure. 
	 * The SimpleMatrix has to be initialized beforehand.
	 * The structure inside the matrix will be as follows: i-th vertex p_i with coordinates (x_ij) will be 
	 * written to M_(i*j+j)_k, where k has to be passed to the method as <colIdx>. 	
	 * @param mesh	The mesh object containing the point list.
	 * @param colIdx	The column index.
	 */
	public void setMeshAtIndex(Mesh mesh, int colIdx){
		assert(colIdx < this.cols) : new IllegalArgumentException("Column index for this mesh is out of bounds.");
		assert((mesh.getPoints().getRows() * dimension) - 1 < this.rows) : new IllegalArgumentException("Row index for this mesh is out of bounds.");
		
		SimpleMatrix points = mesh.getPoints();
		
		for(int i = 0; i < mesh.getPoints().getRows(); i++){
			int rowMajor = dimension * i;
			writePointNdAtIndex(rowMajor, colIdx, points.getRow(i));
		}
	}
	
	/**
	 * Writes the vertex coordinates of a SimpleMatrix into the <colIdx> column of a SimpleMatrix structure. 
	 * The SimpleMatrix has to be initialized beforehand.
	 * The structure inside the matrix will be as follows: i-th row p_i with elements (x_ij) will be 
	 * written to M_(i*j+j)_k, where k has to be passed to the method as <colIdx>. 	
	 * @param mat	The SimpleMatrix.
	 * @param colIdx	The column index.
	 */
	public void setSimpleMatrixAtIndex(SimpleMatrix mat, int colIdx){
		assert(colIdx < this.cols) : new IllegalArgumentException("Column index for this mesh is out of bounds.");
		assert((mat.getRows() * dimension) - 1 < this.rows) : new IllegalArgumentException("Row index for this mesh is out of bounds.");
		
		for(int i = 0; i < mat.getRows(); i++){
			int rowMajor = dimension * i;
			for(int j = 0; j < dimension; j++){
				this.setElementValue(rowMajor+j, colIdx, mat.getElement(i, j));
			}
		}
	}
	
	/**
	 * Sets the connectivity information for the meshes stored in the SimpleMatrix structure.
	 * @param connectivity The indices of the vertices making up the connectivity information.
	 */
	public void setConnectivity(SimpleMatrix connectivity){
		this.connectivity = connectivity;
		this.HAS_CONNECTIVITY = true;
	}
	
	/**
	 * Writes a PointND to <dimension> consecutive row-values starting at a certain index in a certain column.
	 * @param rowMajor	The row starting index.
	 * @param colIdx	The column index.
	 * @param point		The PointND object to be written to the SimpleMatrix structure.
	 */
	private void writePointNdAtIndex(int rowMajor, int colIdx, SimpleVector point){
		for(int i = 0; i < dimension; i++){
			this.setElementValue(rowMajor+i, colIdx, point.getElement(i));
		}
	}
	
	/**
	 * Initialize zero matrix and zero mesh-centers and scaling.
	 * Mesh centers and scalings are only initialized if the class wasn't constructed using a GPA object.
	 * 
	 * @param rows number of rows
	 * @param cols number of columns
	 */
	@Override public void init(final int rows, final int cols) {
		assert (rows >= 0) : new IllegalArgumentException("Number of rows has to be greater than or equal to zero!");
		assert (cols >= 0) : new IllegalArgumentException("Number of columns has to be greater than or equal to zero!");
		if (this.rows != rows || this.cols != cols) {
			this.rows = rows;
			this.cols = cols;
			this.buf = new double[this.rows][this.cols];
			
			if(!USE_GPA){
				this.centers = new ArrayList<PointND>(cols);
				this.scaling = new ArrayList<Float>(cols);
			}
			
		}
	}
	
	/**
	 * Returns the column colIdx of the  {@link DataMatrix} in its original numVertices x dimension {@link SimpleMatrix} form.
	 * @param colIdx The column index where the linearized matrix is stored.
	 * @return SimpleMatrix of the form numVertices x dimension.
	 */
	public SimpleMatrix getSimpleMatrixAtIndex(int colIdx){
		assert(colIdx < this.cols) : new IllegalArgumentException("Column index for this mesh is out of bounds.");
		SimpleMatrix mat = new SimpleMatrix(this.rows/this.dimension, this.dimension);
			
		for(int i = 0; i < mat.getRows(); i++){
			int rowMajor = dimension * i;
			for(int j = 0; j < dimension; j++){
				mat.setElementValue(i, j, this.getElement(rowMajor + j, colIdx));
			}
		}
		return mat;
	}
}

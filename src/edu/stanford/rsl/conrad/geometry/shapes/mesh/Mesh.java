/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.mesh;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.io.VTKMeshIO;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

/**
 * Class to read and store a mesh in a legacy .vtk polydata file. A triangular mesh as produced by Paraview  or the 
 * itkMeshFileWriter is assumed. The reader-method builds upon VTKMeshReader implemented by Marco Boegel. 
 * Extended to meshes featuring deformation vectors at each mesh vertex by Tobias Geimer. 
 * @author Mathias Unberath
 *
 */
public class Mesh{
	
	public static final boolean DEBUG = false;
	
	/**
	 * Dimension of the vertices;
	 */
	public int dimension;
	/**
	 * The number of vertices in the mesh.
	 */
	public int numPoints;
	
	/**
	 * The number of connections in the connectivity information, e.g. triangles.
	 */
	public int numConnections;
	
	/**
	 * The matrix containing the points of the mesh.
	 */
	private SimpleMatrix points;
	/**
	 * The matrix containing the connectivity information.
	 */
	private SimpleMatrix triangles;
	
	/**
	 * The matrix containing the deformation vectors.
	 */
	private SimpleMatrix deformations;
	
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	/**
	 * Constructs an empty Mesh object.
	 */
	public Mesh(){
	}
		
	/**
	 * Constructs a Mesh object and directly calls the readMesh method on the filename input.
	 * @param filename The filename of the mesh to be read.
	 */
	public Mesh(String filename){
		try {
			if(DEBUG){
				System.out.println("Reading mesh in file: " + filename);
			}
				readMesh(filename);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Sets the connectivity and number of connections.
	 * @param con The connectivity.
	 */
	public void setConnectivity(SimpleMatrix con){
		this.triangles = con;
		this.numConnections = con.getRows();
	}
	
	/**
	 * Getter for the connectivity information.
	 * @return The connectivity information.
	 */
	public SimpleMatrix getConnectivity(){
		return this.triangles;
	}
	
	/**
	 * Sets the deformations at each vertex.
	 * @param deform The deformation.
	 */
	public void setDeformation(SimpleMatrix deform){
		this.deformations = deform;
	}
	
	/**
	 * Getter for the deformation information.
	 * @return The deformation information.
	 */
	public SimpleMatrix getDeformation(){
		return this.deformations;
	}
	
	/**
	 * Sets the vertices, e.g points, and number of vertices.
	 * @param p The matrix containing the vertices.
	 */
	public void setPoints(SimpleMatrix p){
		this.points = p;
		this.numPoints = p.getRows();
		this.dimension = p.getCols();
	}
	
	/**
	 * Sets the vertices, e.g points, and number of vertices.
	 * @param p The ArrayList containing the vertices.
	 */
	public void setPoints(ArrayList<PointND> p){
		SimpleMatrix m = toSimpleMatrix(p); 
		this.points = m;
		this.numPoints = m.getRows();
		this.dimension = m.getCols();
	}
	
	/**
	 * Getter for the points, i.e. vertices.
	 * @return The matrix containing the vertices.
	 */
	public SimpleMatrix getPoints(){
		return this.points;
	}
		
	/**
	 * Convenience method to read a triangular mesh in legacy .vtk polydata format using {@link VTKMeshIO}.
	 * @param filename	The filename of the mesh to be read.
	 * @throws IOException if .vtk format does not match expected format
	 */
	public void readMesh(String filename) throws IOException{
		VTKMeshIO reader = new VTKMeshIO(filename);
		reader.read();
		
		Mesh tmpMesh = reader.getMesh();
		this.numPoints = tmpMesh.numPoints;
		this.points = tmpMesh.getPoints();
		this.dimension = tmpMesh.dimension;
		this.numConnections = tmpMesh.numConnections;
		this.triangles = tmpMesh.getConnectivity();
		this.deformations = tmpMesh.getDeformation();
	}

	/**
	 * Converts the ArrayList into a SimpleMatrix structure.
	 * @param list The ArrayList to be converted.
	 * @return	The SimpleMatrix containing the ArrayList's entries.
	 */
	private SimpleMatrix toSimpleMatrix(ArrayList<PointND> list){
		int rows = list.size();
		int cols = list.get(0).getDimension();
		SimpleMatrix pts = new SimpleMatrix(rows,cols);
		
		for(int i = 0; i < rows; i++){
			PointND point = list.get(i);
			for(int j = 0; j < cols; j++){
				pts.setElementValue(i, j, point.get(j));
			}
		}
		return pts;
	}
	
}
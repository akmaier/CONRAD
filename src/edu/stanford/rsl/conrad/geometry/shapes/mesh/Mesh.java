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
	 * Getter for the points, i.e. vertices.
	 * @return The matrix containing the vertices.
	 */
	public SimpleMatrix getPoints(){
		return this.points;
	}
		
	/**
	 * Method to read a triangular mesh in legacy .vtk polydata format. 
	 * @param filename	The filename of the mesh to be read.
	 * @throws IOException if .vtk format does not match expected format
	 */
	public void readMesh(String filename) throws IOException{
		
		ArrayList<PointND> points = new ArrayList<PointND>();
		ArrayList<PointND> triangles = new ArrayList<PointND>();
		ArrayList<PointND> deformations = new ArrayList<PointND>();
		
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);

		// read and discard header information
		br.readLine();
		br.readLine();
		br.readLine();
		br.readLine();

		String line = br.readLine();
		StringTokenizer tok = new StringTokenizer(line);
		String t = tok.nextToken(); // skip "points"
		t = tok.nextToken();
		int numPoints = Integer.parseInt(t);
				
		// read points
		// the logic here allows more than one single point per line
		for (int i = 0; i < numPoints;){
			line = br.readLine();
			tok = new StringTokenizer(line);
			int nrPts = tok.countTokens();
			nrPts /= 3;
			for (int j = 0; j < nrPts; j++){
				PointND p = new PointND(Float.parseFloat(tok.nextToken()), Float.parseFloat(tok.nextToken()), Float.parseFloat(tok.nextToken()));
				points.add(p);
			}
			i += nrPts;
		}
		// read connectivity information
		// assumes triangle mesh, hence first number in connectivity information needs to be 3
		// logic allows more than one triangle per line
		line = br.readLine();
		if(line.isEmpty()){
			line = br.readLine();
		}
		tok = new StringTokenizer(line);
		tok.nextToken(); // skip "polygons"
		int numTri =  Integer.parseInt(tok.nextToken());
		for(int i = 0; i < numTri;){
			line = br.readLine();
			tok = new StringTokenizer(line);
			int nTri = tok.countTokens();
			nTri /= 4;
			for(int j = 0; j < nTri; j++){
				t = tok.nextToken();
				if(Integer.parseInt(t) != 3){
					br.close();
					fr.close();
					throw new IOException("VTK-Polydata file: Format not yet supported.");
				}else{
					PointND triangle = new PointND(Integer.parseInt(tok.nextToken()),Integer.parseInt(tok.nextToken()),Integer.parseInt(tok.nextToken()));
					triangles.add(triangle);
				}
			}
			i += nTri;
		}
		
		// read deformation vectors
		boolean hasDeform = true;
		
		line = br.readLine();
		if(line!=null && line.isEmpty()){
			line = br.readLine();
		}
		// not every mesh features a deformation field
		// check whether the file contains any more information
		if(line == null) {
			hasDeform = false; 
		} else {
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Point_Data"
			int numVec =  Integer.parseInt(tok.nextToken());
			line = br.readLine(); // skip "Field FieldData 6"
			line = br.readLine(); // skip "Phi_Glyph numComp numTuples dataType"
			for(int i = 0; i < numVec;){
				line = br.readLine();
				tok = new StringTokenizer(line);
				int nVec = tok.countTokens();
				nVec /= 3;
				for(int j = 0; j < nVec; j++){
					PointND vector = new PointND(Float.parseFloat(tok.nextToken()),Float.parseFloat(tok.nextToken()),Float.parseFloat(tok.nextToken()));
					deformations.add(vector);
				}
				i += nVec;
			}
		}		
		br.close();	
		fr.close();
		
		this.numPoints = numPoints;
		this.points = toSimpleMatrix(points);
		this.dimension = this.points.getCols();
		this.numConnections = numTri;
		this.triangles = toSimpleMatrix(triangles);
		this.deformations = hasDeform ? toSimpleMatrix(deformations) : null;
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
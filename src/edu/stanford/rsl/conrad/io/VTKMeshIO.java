/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;


/**
 * Class to write legacy .vtk polydata meshes to a file. 
 * Only triangular meshes are supported at the moment.
 * @author Mathias Unberath
 *
 */
public class VTKMeshIO {
	
	/**
	 * The filename of the output file.
	 */
	public String filename;
	
	/**
	 * The mesh file to be written to <filename>.
	 */
	public Mesh mesh;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================\
	
	/**
	 * Class to write legacy .vtk polydata files.
	 */
	public VTKMeshIO(){
		
	}
	
	/**
	 * Constructs a writer object and sets the output filename.
	 * @param filename The output file.
	 */
	public VTKMeshIO(String filename){
		this.filename = filename;
	}
	
	/**
	 * Constructs a writer object and sets the output filename and mesh to be written.
	 * @param filename The output file.
	 * @param mesh	The mesh to be written.
	 */
	public VTKMeshIO(String filename, Mesh mesh){
		this.filename = filename;
		this.mesh = mesh;
	}

	/**
	 * Sets the output filename.
	 * @param filename The output file.
	 */
	public void setFilename(String filename){
		this.filename = filename;
	}
	
	/**
	 * Sets the mesh to be written.
	 * @param filename The mesh to be written.
	 */
	public void setMesh(Mesh mesh){
		this.mesh = mesh;
	}
	
	/**
	 * Getter for the mesh.
	 * @return The mesh.
	 */
	public Mesh getMesh(){
		return this.mesh;
	}
	
	/**
	 * Write the mesh to a file.
	 */
	public void write(){
		assert(filename != null) : new Exception("Filename has not been set.");
		assert(mesh != null) : new Exception("Mesh has not been set.");
		
		System.out.println("Writing file: " + filename);
		
		SimpleMatrix points = mesh.getPoints();
		SimpleMatrix triangles = mesh.getConnectivity();
		
		if(points == null){
			points = new SimpleMatrix(0,0);
		}
		if(triangles == null){
			triangles = new SimpleMatrix(0,0);
		}
		
		try {
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			//write header information
			writer.println("# vtk DataFile Version 3.0");
			writer.println("vtk output");
			writer.println("ASCII");
			writer.println("DATASET POLYDATA");
			// write number of points and then each point in one line
			writer.println("POINTS "+points.getRows()+" float");
			for(int i = 0; i < points.getRows(); i++){
				writer.println(points.getElement(i, 0)+" "+points.getElement(i, 1)+" "+points.getElement(i, 2));
			}
			// write number of triangles, number of total entries and then each triangle in one line
			writer.println("POLYGONS "+triangles.getRows()+" "+4*triangles.getRows());
			for(int i = 0; i < triangles.getRows(); i++){
				writer.println("3 " + Integer.toString((int)triangles.getElement(i, 0)) + " " 
									+ Integer.toString((int)triangles.getElement(i, 1)) + " " 
									+ Integer.toString((int)triangles.getElement(i, 2)));
			}
			writer.println();
			
			writer.close();
			
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	
	/**
	 * Method to read a triangular mesh in legacy .vtk polydata format. 
	 * @throws IOException if .vtk format does not match expected format
	 */
	public void read() throws IOException{
		
		ArrayList<PointND> points = new ArrayList<PointND>();
		ArrayList<PointND> triangles = new ArrayList<PointND>();
		
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
		
		br.close();	
		fr.close();
		
		this.mesh = new Mesh();
		this.mesh.numPoints = numPoints;
		this.mesh.setPoints(toSimpleMatrix(points));
		this.mesh.dimension = this.mesh.getPoints().getCols();
		this.mesh.numConnections = numTri;
		if(triangles.size()!=0) this.mesh.setConnectivity(toSimpleMatrix(triangles));
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

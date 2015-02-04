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
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.PCA;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


/**
 * Class to read and write Eigenvectors, Eigenvalues and the consensus as produced by a principal component analysis.
 * @author Mathias Unberath
 *
 */
public class PcaIO {

	/**
	 * Dimension of vertices in a mesh object. Default is 1;
	 */
	private int pointDimension = 1;
	
	/**
	 * The filename for input or output operation.
	 */
	String filename;
	
	/**
	 * Array containing the eigenvalues of the covariance matrix after singular value decomposition.
	 */
	private double[] eigenValues;
	
	/**
	 * Matrix containing the Eigenvectors of the covariance matrix after singular value decomposition.
	 */
	private SimpleMatrix eigenVectors;
	
	/**
	 * Connectivity information when dealing with meshes.
	 */
	private SimpleMatrix connectivity;
	
	/**
	 * The mean shape.
	 */
	private SimpleVector consensus;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	/**
	 * Default constructor.
	 */
	public PcaIO(){
		
	}
	
	/**
	 * Constructs the object and sets the filename for reading operation.
	 * @param filename The file to be read.
	 */
	public PcaIO(String filename){
		this.filename = filename;
	}
	
	/**
	 * Reads the variances from the file in the filename class member and stores the data in the corresponding class members.
	 * @throws IOException 
	 */
	public void readVarianceOnly() throws IOException{
		assert(filename != null) : new Exception("Filename not set.");
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);
		
		String line = br.readLine();
		StringTokenizer tok = new StringTokenizer(line);
		tok.nextToken(); // skip "DIMENSION"
		String t = tok.nextToken();	
		@SuppressWarnings("unused")
		int rows = Integer.parseInt(t);
		t = tok.nextToken();
		int cols = Integer.parseInt(t);
		// read point dimension
		line = br.readLine();
		tok = new StringTokenizer(line);
		tok.nextToken(); // skip "POINTDIMENSION"
		this.pointDimension = Integer.parseInt(tok.nextToken());
		
		// allocate class members
		this.eigenValues = new double[cols];
		
		br.readLine(); // skip "EIGENVALUES"
		line = br.readLine(); // read eigenvalues
		tok = new StringTokenizer(line);
		for(int i = 0; i < cols; i++){
			eigenValues[i] = Double.parseDouble(tok.nextToken());
		}
		br.close();
		fr.close();
	}
	
	
	/**
	 * Reads the file in the filename class member and stores the data in the corresponding class members.
	 * @throws IOException 
	 */
	public void readFile() throws IOException{
		assert(filename != null) : new Exception("Filename not set.");
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);
		
		String line = br.readLine();
		StringTokenizer tok = new StringTokenizer(line);
		tok.nextToken(); // skip "DIMENSION"
		String t = tok.nextToken();	
		int rows = Integer.parseInt(t);
		t = tok.nextToken();
		int cols = Integer.parseInt(t);
		// read point dimension
		line = br.readLine();
		tok = new StringTokenizer(line);
		tok.nextToken(); // skip "POINTDIMENSION"
		this.pointDimension = Integer.parseInt(tok.nextToken());
		
		// allocate class members
		this.eigenValues = new double[cols];
		this.eigenVectors = new SimpleMatrix(rows, cols);
		this.consensus =  new SimpleVector(rows);
		
		br.readLine(); // skip "EIGENVALUES"
		line = br.readLine(); // read eigenvalues
		tok = new StringTokenizer(line);
		for(int i = 0; i < cols; i++){
			eigenValues[i] = Double.parseDouble(tok.nextToken());
		}
		br.readLine(); // skip "EIGENVECTORS | CONSENSUS"
		for(int i = 0; i < rows; i++){
			line = br.readLine();
			tok = new StringTokenizer(line);
			for(int j = 0; j < cols; j++){
				eigenVectors.setElementValue(i, j, Double.parseDouble(tok.nextToken()));
			}
			consensus.setElementValue(i, Double.parseDouble(tok.nextToken())); // last entry is consensus
		}
		// try to read connectivity if exists
		line = br.readLine();
		if(line != null){
			tok = new StringTokenizer(line);
			tok.nextToken();
			this.connectivity = new SimpleMatrix(Integer.parseInt(tok.nextToken()),Integer.parseInt(tok.nextToken()));
			for(int i = 0; i < connectivity.getRows(); i++){
				line = br.readLine();
				tok = new StringTokenizer(line);
				tok.nextToken(); // skip leading number of connections
				for(int j = 0; j < connectivity.getCols(); j++){
					connectivity.setElementValue(i, j, Double.parseDouble(tok.nextToken()));
				}
			}
		}
			
		br.close();
		fr.close();
		
	}
	
	/**
	 * Writes the data in the class members to the file specified in filename.
	 */
	public void writeFile(){
		assert( filename != null) : new Exception("Filename not set.");
		assert( consensus != null && eigenVectors != null && eigenValues != null) : new Exception("Data not set.");
		
		try {
			System.out.println("Writing Eigen-Values and Eigen-Vectors to file: " + filename);
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			writer.println("DIMENSION " + this.eigenVectors.getRows() + " " + this.eigenVectors.getCols());
			writer.println("POINTDIMENSION " + this.pointDimension);
			writer.println("EIGENVALUES");
			writer.println(buildStringEigenValues());
			writer.println("EIGENVECTORS | CONSENSUS");
			for(int i = 0; i < this.eigenVectors.getRows(); i++){
				writer.println(buildStringEigenVectorAtIndex(i) + consensus.getElement(i));
			}
			//write connectivity if exists
			if(connectivity != null){
				writer.println("CONNECTIVITY " + connectivity.getRows() + " " +connectivity.getCols());
				for(int i = 0; i < connectivity.getRows(); i++){
					writer.println(buildStringConnectivity(i));
				}
			}
			writer.close();
			System.out.println("Finished writing.");
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Constructs the string object of connectivity-entries in the row specified for writing operation.
	 * @param idx The row index.
	 * @return The string containing the connectivity row.
	 */
	private String buildStringConnectivity(int idx){
		StringBuilder ev = new StringBuilder();
		ev.append(connectivity.getCols());
		for(int i = 0; i < connectivity.getCols(); i++){
			ev.append(" ");
			ev.append(connectivity.getElement(idx, i));
		}
		return ev.toString();
	}
	
	/**
	 * Constructs the string object of Eigenvector-entries in the row specified for writing operation.
	 * @param idx The row index.
	 * @return The string containing the Eigenvector row.
	 */
	private String buildStringEigenVectorAtIndex(int idx){
		StringBuilder ev = new StringBuilder();
		for(int i = 0; i < eigenValues.length; i++){
			ev.append(eigenVectors.getElement(idx, i));
			ev.append(" ");
		}
		return ev.toString();
	}
	
	/**
	 * Constructs the string object of Eigenvalues for writing operation.
	 * @return The string containing the Eigenvalues.
	 */
	private String buildStringEigenValues(){
		StringBuilder ev = new StringBuilder();
		for(int i = 0; i < eigenValues.length; i++){
			ev.append(eigenValues[i]);
			ev.append(" ");
		}
		return ev.toString();
	}
	
	/**
	 * Sets the filename for IO operation.
	 * @param filename The file to be read or written.
	 */
	public void setFilename(String filename){
		this.filename = filename;
	}
	
	/**
	 * Constructs the object and sets Eigenvalues, Eigenvectors and the consensus for writing operation.
	 * @param eVal
	 * @param eVec
	 * @param mean
	 */
	public PcaIO(int pointDim, double[] eVal, SimpleMatrix eVec, SimpleVector mean){
		this.pointDimension = pointDim;
		this.eigenValues = eVal;
		this.eigenVectors = eVec;
		this.consensus = mean;
	}
	
	
	/**
	 * Constructs the object and sets the filename, point dimension, eigenvalues, eigenvectors, mean shape and connectivity for writing.
	 * @param filename
	 * @param pca
	 */
	public PcaIO(String filename, PCA pca){
		this.filename = filename;
		this.pointDimension = pca.dimension;
		this.eigenValues = pca.eigenValues;
		this.eigenVectors = pca.eigenVectors;
		this.consensus = pca.getConsensus();
		this.connectivity = pca.connectivity;
	}
	
	/**
	 * Getter for the dimension of points. Needed if a point-cloud is analyzed using PCA.
	 * @return The point dimension.
	 */
	public int getPointDimension(){
		return this.pointDimension;
	}
	
	/**
	 * Getter for the Eigenvalues after reading operation.
	 * @return The Eigenvalues.
	 */
	public double[] getEigenValues(){
		return this.eigenValues;
	}
	
	/**
	 * Getter for the Eigenvectors after reading operation.
	 * @return The Eigenvectors.
	 */
	public SimpleMatrix getEigenVectors(){
		return this.eigenVectors;
	}
	
	/**
	 * Getter for the connecivity after reading operation.
	 * @return The connectivity.
	 */
	public SimpleMatrix getConnectivity(){
		return this.connectivity;
	}
	
	/**
	 * Getter for the consensus object after reading operation.
	 * @return The consensus object.
	 */
	public SimpleVector getConsensus(){
		return this.consensus;
	}
}

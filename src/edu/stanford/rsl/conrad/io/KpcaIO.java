/*
 * Copyright (C) 2010-2014 Mathias Unberath CONRAD is developed as an Open
 * Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.io;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.KPCA;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels.GaussianKernel;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels.PolynomialKernel;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.DataMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to read and write Eigenvectors, Eigenvalues and the training-sets as
 * used in kernel principal component analysis.
 * 
 * @author Mathias Unberath
 * 
 */
public class KpcaIO {

	/**
	 * Dimension of vertices in a mesh object. Default is 1;
	 */
	private int pointDimension = 1;

	/**
	 * The filename for input or output operation.
	 */
	String filename;

	/**
	 * Array containing the eigenvalues of the covariance matrix after singular
	 * value decomposition.
	 */
	private double[] eigenValues;

	/**
	 * Matrix containing the Eigenvectors of the covariance matrix after
	 * singular value decomposition.
	 */
	private SimpleMatrix eigenVectors;

	/**
	 * Training-sets.
	 */
	private SimpleMatrix trainingSets;

	/**
	 * Feature Matrix K.
	 */
	private SimpleMatrix featureMatrix;

	/**
	 * Consensus object needed to project new shapes.
	 */
	private SimpleVector consensus;
	
	/**
	 * The name of the kernel.
	 */
	private String kernel;
	
	/**
	 * The variation threshold used to reduce dimensionality.
	 */
	private double variation;

	// ==========================================================================================
	// METHODS
	// ==========================================================================================

	/**
	 * Constructor for KPCA writing operation. 
	 * @param filename
	 * @param kpca
	 */
	public KpcaIO(String filename, KPCA kpca) {
		this.filename = filename;
		this.pointDimension = kpca.dimension;
		this.eigenValues = kpca.eigenValues;
		this.eigenVectors = kpca.eigenVectors; // the eigenvectors are commonly referred to as alpha
		this.trainingSets = kpca.data;
		this.featureMatrix = kpca.getFeatureMatrix();
		this.consensus = toSimpleVector(kpca.data.consensus);
		this.kernel = kpca.kernel.getName();
		this.variation =  kpca.variationThreshold;
	}

	/**
	 * Constructor for reading operation. 
	 * @param filename The filename containing the Kpca Object as produced by this class
	 */
	public KpcaIO(String filename){
		this.filename = filename;
	}
	
	/**
	 * Writes the previously set KPCA object to the specified file.
	 */
	public void writeFile(){
		assert( filename != null) : new Exception("Filename not set.");
		assert( consensus != null && eigenVectors != null && eigenValues != null && trainingSets != null && featureMatrix != null && kernel != null) 
			: new Exception("Data not set.");
		
		System.out.println("Writing to file: " + filename);
		
		try{
			System.out.println("Writing Eigen-Values and Eigen-Vectors to file: " + filename);
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			
			writer.println("Number of datasets: " + Integer.valueOf(trainingSets.getCols()));
			writer.println("Number of points: " + Integer.valueOf(trainingSets.getRows()/pointDimension));
			writer.println("Point dimension: " + Integer.valueOf(pointDimension));
			writer.println("Number of P.C.s: " + Integer.valueOf(eigenValues.length));
			writer.println("Variation threshold: " + Double.valueOf(variation));
			writer.println("Kernel: " + kernel);
			
			writer.println("");
			writer.println("EIGENVALUES of K_ij");
			String eval = "";
			for(int i = 0; i < eigenValues.length; i++){
				eval += (Double.valueOf(eigenValues[i]) + " ");
			}
			writer.println(eval);
			
			writer.println("EIGENVECTORS of K_ij");
			for(int i = 0; i < trainingSets.getCols(); i++){
				writer.println(rowAsString(eigenVectors, i));
			}
			
			writer.println("FEATUREMATRIX K_ij");
			for(int i = 0; i < trainingSets.getCols(); i++){
				writer.println(rowAsString(featureMatrix, i));
			}
			
			writer.println("TRAININGSETS | CONSENSUS");
			for(int i = 0; i < trainingSets.getRows(); i++){
				writer.println(rowAsString(trainingSets, i) + Double.valueOf(consensus.getElement(i)));
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Constructs a KPCA object from the data given in the file referred to by filename.
	 * @return KPCA object
	 */
	public KPCA readFile(){
		assert(filename != null) : new Exception("Filename not set.");
		
		KPCA kpca= new KPCA();
				
		FileReader fr;
		try {
			fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Number of Datasets"
			tok.nextToken();
			tok.nextToken();
			String t = tok.nextToken();
			kpca.numSamples = Integer.parseInt(t);
			
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Number of points"
			tok.nextToken();
			tok.nextToken();
			t = tok.nextToken();
			kpca.numVertices = Integer.parseInt(t);
			
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Point dimension"
			tok.nextToken();
			t = tok.nextToken();
			kpca.dimension = Integer.parseInt(t);
			kpca.numPoints = kpca.numVertices * kpca.dimension;
			
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Number of Pcs"
			tok.nextToken();
			tok.nextToken();
			t = tok.nextToken();
			int numPC = Integer.parseInt(t);
			
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Variation threshold"
			tok.nextToken();
			t = tok.nextToken();
			kpca.variationThreshold = Double.parseDouble(t);
			
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "Kernel"
			t = tok.nextToken();
			String kernelName = t;
			t = tok.nextToken();
			double val = Double.parseDouble(t);
			if(kernelName.equals("Gaussian")){
				kpca.kernel = new GaussianKernel(val);
			}else if(kernelName.equals("Polynomial")){
				double alpha = Double.valueOf(tok.nextToken());
				double offs = Double.valueOf(tok.nextToken());
				kpca.kernel = new PolynomialKernel((int) val, alpha, offs);
			}else{
				System.out.println("Kernel method unknown. Using default: parabolic kernel.");
				kpca.kernel = new PolynomialKernel();
			}
			// read the EIGENVALUES
			br.readLine();
			br.readLine(); // skip empty line, then skip "EIGENVALUES ..."
			line = br.readLine();
			tok = new StringTokenizer(line);
			double[] ev = new double[numPC];
			for(int i = 0; i < numPC; i++){
				t = tok.nextToken();
				ev[i] = Double.parseDouble(t);
			}
			kpca.eigenValues = ev;
			
			// read the EIGENVECTORS
			br.readLine(); // skip "EIGENVECTORS ..."
			SimpleMatrix evec = new SimpleMatrix(kpca.numSamples, numPC);
			for(int i = 0; i < kpca.numSamples; i++){
				line = br.readLine();
				tok = new StringTokenizer(line);
				for(int j = 0; j < numPC; j++){
					t = tok.nextToken();
					evec.setElementValue(i, j, Double.parseDouble(t));
				}
			}
			kpca.eigenVectors = evec;
			
			// read the Feature Matrix
			br.readLine(); // skip "FEATUREMATRIX ..."
			SimpleMatrix feat = new SimpleMatrix(kpca.numSamples, kpca.numSamples);
			for(int i = 0; i < kpca.numSamples; i++){
				line = br.readLine();
				tok = new StringTokenizer(line);
				for(int j = 0; j < kpca.numSamples; j++){
					t = tok.nextToken();
					feat.setElementValue(i, j, Double.parseDouble(t));
				}
			}
			kpca.setFeatureMatrix(feat);
			
			// read the trainingssets and consensus
			br.readLine(); // skip "FEATUREMATRIX ..."
			DataMatrix m = new DataMatrix();
			m.init(kpca.numPoints, kpca.numSamples);
			m.dimension = kpca.dimension;
			
			SimpleMatrix consensus = new SimpleMatrix(kpca.numVertices, kpca.dimension);
			for(int i = 0; i < kpca.numPoints; i++){
				int idxx = (int)Math.floor((float)i/kpca.dimension);
				int idxy = i - idxx * kpca.dimension;
				SimpleVector vec = new SimpleVector(kpca.numSamples);
				line = br.readLine();
				tok = new StringTokenizer(line);
				for(int j = 0; j < kpca.numSamples; j++){
					t = tok.nextToken();
					if(j == kpca.numSamples-1){
						consensus.setElementValue(idxx, idxy, Double.parseDouble(t));
					}else{
						vec.setElementValue(j, Double.parseDouble(t));
					}
				}
			}
			m.consensus = consensus;
			kpca.data = m;
			
			br.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		return kpca;
	}
	
	
	
	
	/**
	 * Condenses the i_th row of a simple matrix into a string. 
	 * @param m
	 * @param row
	 * @return i_th row as string
	 */
	private String rowAsString(SimpleMatrix m, int row){
		String r = "";
		for(int i = 0; i < m.getCols(); i++){
			r += (Double.valueOf(m.getElement(row, i)) + " ");
		}
		return r;
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
}


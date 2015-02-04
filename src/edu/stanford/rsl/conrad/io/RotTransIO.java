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

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Class to read a 3x3 rotation matrix and a translation vector stored in a file.
 * The reader assumes a 3x4 space separated structure containing the translation vector as the 4th column.
 * @author Mathias Unberath
 *
 */
public class RotTransIO {

	/**
	 * The file to be read.
	 */
	private String filename;
	
	private SimpleMatrix rotation;
	
	private SimpleVector translation;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
		
	public RotTransIO(String filename){
		this.filename = filename;
		try {
			read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public SimpleMatrix getAffineMapping(){
		SimpleMatrix out = new SimpleMatrix(4,4);
		out.setSubMatrixValue(0, 0, rotation);
		out.setSubColValue(0, 3, translation);
		out.setElementValue(3, 3, 1);
		return out;
	}
	
	public SimpleMatrix getRotation(){
		return this.rotation;
	}
	
	public SimpleVector getTranslation(){
		return this.translation;
	}
	
	public void newFile(String filename){
		this.filename = filename;
		try {
			read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public RotTransIO(String filename, SimpleMatrix rot, SimpleVector trans){
		this.filename = filename;
		write(rot, trans);
	}
	
	/**
	 * Read the rotation and translation in filename.
	 * @throws IOException
	 */
	private void read() throws IOException{
		SimpleMatrix rotation = new SimpleMatrix(3,3);
		SimpleVector translation = new SimpleVector(3);
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);

		String line = br.readLine(); // skip first line
		StringTokenizer tok;
				
		// read rotation matrix 3x3 and translation 3x1
		for (int i = 0; i < 3; i++){
			line = br.readLine();
			tok = new StringTokenizer(line);
			SimpleVector row = new SimpleVector(Float.parseFloat(tok.nextToken()), Float.parseFloat(tok.nextToken()), Float.parseFloat(tok.nextToken()));
			rotation.setRowValue(i, row);
			translation.setElementValue(i, Float.parseFloat(tok.nextToken()));
		}
		
		br.close();
		fr.close();
		
		this.rotation = rotation;
		this.translation = translation;
				
	}
	
	private void write(SimpleMatrix rot, SimpleVector trans){
		assert(filename != null && rot.getCols() == 3 && rot.getRows() == 3 && trans.getLen() == 3) : new Exception();
		try {
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			writer.println("ROTATION | TRANSLATION");
			for(int i = 0; i < rot.getRows(); i++){
				String line = "";
				for( int j = 0; j < rot.getCols() + 1; j++){
					line += " ";
					line += (j != rot.getCols()) ? rot.getElement(i, j):trans.getElement(i);
				}
				writer.println(line);
			}
			writer.close();			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
	
}
